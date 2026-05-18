/**
 * Web Worker hosting the ONNX encoder + decoder.
 *
 * Why a worker: terminating the worker is the only browser mechanism that
 * actually returns WASM linear memory to the OS. The main thread keeps the
 * worker alive for one batch of attacks, then `worker.terminate()` releases
 * everything — heap, model weights, session caches.
 *
 * Message protocol (id-correlated):
 *   { id, type: 'init',   encoderBuf, decoderBuf }
 *   { id, type: 'encode', imageBuf, W, H, bitsBuf, permBuf, permP }
 *   { id, type: 'decode', imageBuf, W, H,           permBuf, permP }
 *
 * `permBuf` is an Int32Array buffer of length 12*1024*permP, CHW perm stack
 * computed by the main thread via assets/imagehide/perm.js. The ONNX graph
 * takes the perm as an input tensor so it's truly size-polymorphic instead
 * of using baked indices from one trace size.
 *
 * Replies:
 *   { id, type: 'ready'  }
 *   { id, type: 'encoded', f32Buf,   ms }   // Float32Array CHW, [-1, 1]
 *   { id, type: 'decoded', bitsBuf,  ms }
 *   { id, type: 'error',   message }
 *
 * All large payloads (imageBuf, bitsBuf) cross the boundary as ArrayBuffer
 * transferables to avoid copies.
 */

let ort = null;
let session = null;          // single active session — either encoder or decoder
let sessionMode = null;      // 'encoder' | 'decoder'
let activeBackend = 'wasm';

const ORT_VERSION = '1.20.0';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
// WebGPU bundle includes the WASM provider as fallback.
const ORT_BUNDLE = `${ORT_BASE}ort.webgpu.min.mjs`;

// Mirror worker diagnostics to the main thread so they show in the
// page's devtools console even when the worker context is hidden
// (Safari/Firefox default behavior). Tagged so the deploy-version is
// visible in the worker logs too, helping diagnose stale-cache cases.
const _DIAG_TAG = '[v2]';
function diag(msg) {
  // Single channel: postMessage to main thread, which logs once. We
  // deliberately do NOT also console.log here — Safari shows worker
  // console output as a separate file context, which doubled every line
  // in the user-facing console. postMessage is reliable enough that the
  // worker-side log was redundant.
  try { self.postMessage({ id: 0, type: 'diag', message: `${_DIAG_TAG} ${msg}` }); } catch (_) {}
}

function isIOS() {
  // iPhone/iPad/iPod (all WebKit). Used to pick a WebGPU-only provider list
  // on iOS so a partial WebGPU init can't poison the shared WASM init in
  // the same session. macOS Safari is stable with the mixed provider list.
  const ua = (typeof navigator !== 'undefined' && navigator.userAgent) || '';
  return /iPhone|iPad|iPod/i.test(ua);
}

async function detectWebGPU() {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) return false;
  // iOS skip: tried multiple times to make WebGPU usable on iPhone. The
  // attempt itself reserves GPU resources that iOS doesn't release even on
  // worker.terminate(), starving the WASM fallback. Symptom: encode/decode
  // fails with "no available backend found. ERR: [wasm] RangeError: Out of
  // memory" even though the page had plenty of headroom on entry. macOS
  // Safari is unaffected — keep WebGPU there.
  if (isIOS()) {
    diag('WebGPU: skipping on iOS (pins GPU memory, breaks WASM fallback)');
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      diag('WebGPU: requestAdapter returned null');
      return false;
    }
    const info = adapter.info || {};
    diag(`WebGPU adapter: vendor=${info.vendor || '?'} arch=${info.architecture || '?'} device=${info.device || '?'}`);
    return true;
  } catch (e) {
    diag(`WebGPU detect failed: ${e.message}`);
    return false;
  }
}

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(
      () => reject(new Error(`${label} timed out after ${ms}ms`)), ms)),
  ]);
}

function disposeTensor(t) {
  if (t && typeof t.dispose === 'function') {
    try { t.dispose(); } catch (_) { /* ignore */ }
  }
}

function imageBufToFloat32CHW(buf, W, H) {
  const data = new Uint8ClampedArray(buf);
  const out = new Float32Array(3 * H * W);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const pi = y * W + x;
      out[pi]              = (data[i]     / 127.5) - 1;
      out[H * W + pi]      = (data[i + 1] / 127.5) - 1;
      out[2 * H * W + pi]  = (data[i + 2] / 127.5) - 1;
    }
  }
  return out;
}

// (Removed float32CHWtoImageBuf — main thread now handles f32 → uint8 conversion
// so we can keep the float32 container alive for the attack pipeline. The worker
// returns the raw clamped Float32 CHW buffer.)

async function handle(msg) {
  if (msg.type === 'init') {
    diag(`init received for mode=${msg.mode}${msg.forceWasm ? ' forceWasm=true' : ''}`);
    if (!ort) {
      ort = await import(ORT_BUNDLE);
      if (ort.env && ort.env.wasm) ort.env.wasm.wasmPaths = ORT_BASE;
    }
    const { mode, modelBuf, forceWasm } = msg;
    const bytes = new Uint8Array(modelBuf);
    const tryWebGPU = !forceWasm && (await detectWebGPU());
    if (tryWebGPU) {
      try {
        // iOS: webgpu-only (no WASM in providers list). When WebGPU
        // partially fails inside a mixed list, ORT half-initializes WASM
        // and caches an init failure — which then kills the wasm-only
        // respawn too. With webgpu-only, a failure leaves WASM untouched
        // and pipeline.js's respawn-as-forceWasm starts clean.
        // macOS / non-iOS: keep mixed list (some ops benefit from the
        // WASM fallback inside the same session, e.g. shape ops).
        const providers = isIOS() ? ['webgpu'] : ['webgpu', 'wasm'];
        const t0 = performance.now();
        session = await withTimeout(
          ort.InferenceSession.create(bytes, { executionProviders: providers }),
          10000, `${mode} webgpu init`);
        diag(`WebGPU session ready for ${mode} (providers=${providers.join(',')}) in ${(performance.now()-t0).toFixed(0)}ms`);
        sessionMode = mode;
        activeBackend = 'webgpu';
        return { type: 'ready', backend: activeBackend, transfer: [] };
      } catch (e) {
        diag(`WebGPU session create failed for ${mode}: ${e.message}`);
        try { session?.release?.(); } catch (_) {}
        session = null;
      }
    }
    diag(`Using WASM backend for ${mode}`);
    session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    sessionMode = mode;
    activeBackend = 'wasm';
    return { type: 'ready', backend: activeBackend, transfer: [] };
  }

  if (msg.type === 'encode') {
    if (sessionMode !== 'encoder') throw new Error('worker not in encoder mode');
    const { imageBuf, W, H, bitsBuf, permBuf, permP } = msg;
    const imgArr  = imageBufToFloat32CHW(imageBuf, W, H);
    const bitsArr = new Float32Array(bitsBuf);
    // perm_stack is Int32 CHW-perm of shape (12, 1024, p) but ONNX expects
    // int64 indices for ScatterElements. Read bytes directly via DataView
    // so we never hold a separate Int32Array view alive alongside the
    // BigInt64Array — saves ~6 MB transient at 1 MP. Little-endian matches
    // the JS engine's typed-array byte order on all common platforms.
    const permLen = (permBuf.byteLength >>> 2);
    const permI64 = new BigInt64Array(permLen);
    {
      const dv = new DataView(permBuf);
      for (let i = 0; i < permLen; i++) permI64[i] = BigInt(dv.getInt32(i << 2, true));
    }
    const imgT  = new ort.Tensor('float32', imgArr,  [1, 3, H, W]);
    const bitsT = new ort.Tensor('float32', bitsArr, [1, bitsArr.length]);
    const permT = new ort.Tensor('int64',   permI64, [12, 1024, permP]);
    const t0 = performance.now();
    const { container_rgb } = await session.run({
      host_rgb: imgT, bits: bitsT, perm_stack: permT });
    const ms = performance.now() - t0;
    // Copy the Float32 CHW data out of ORT-owned session memory (clamped to
    // [-1, 1] to remove tiny overshoots), then transfer the buffer to the
    // main thread. Main thread converts to uint8 ImageData for display and
    // keeps the float32 version for the attack pipeline so sub-uint8 residual
    // survives the resize step.
    const src = container_rgb.data;
    const f32Out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) {
      const v = src[i];
      f32Out[i] = v < -1 ? -1 : (v > 1 ? 1 : v);
    }
    disposeTensor(imgT); disposeTensor(bitsT); disposeTensor(permT);
    disposeTensor(container_rgb);
    return { type: 'encoded', f32Buf: f32Out.buffer, ms, transfer: [f32Out.buffer] };
  }

  if (msg.type === 'decode') {
    if (sessionMode !== 'decoder') throw new Error('worker not in decoder mode');
    const { imageBuf, W, H, permBuf, permP } = msg;
    const arr = imageBufToFloat32CHW(imageBuf, W, H);
    // Same Int32→Int64 widening trick as encode: read via DataView so we
    // never hold both views in memory at once.
    const permLen = (permBuf.byteLength >>> 2);
    const permI64 = new BigInt64Array(permLen);
    {
      const dv = new DataView(permBuf);
      for (let i = 0; i < permLen; i++) permI64[i] = BigInt(dv.getInt32(i << 2, true));
    }
    const t     = new ort.Tensor('float32', arr,     [1, 3, H, W]);
    const permT = new ort.Tensor('int64',   permI64, [12, 1024, permP]);
    const t0 = performance.now();
    const { bit_logits } = await session.run({ container_rgb: t, perm_stack: permT });
    const ms = performance.now() - t0;
    const src = bit_logits.data;
    const bits = new Uint8Array(src.length);
    for (let i = 0; i < bits.length; i++) {
      const s = 1 / (1 + Math.exp(-src[i]));
      bits[i] = s > 0.5 ? 1 : 0;
    }
    disposeTensor(t); disposeTensor(permT); disposeTensor(bit_logits);
    return { type: 'decoded', bitsBuf: bits.buffer, ms, transfer: [bits.buffer] };
  }

  if (msg.type === 'release') {
    // Cleanly release ORT session resources before the worker is terminated.
    // ORT's session.release() flushes any in-flight WebGPU command buffers and
    // frees device-side buffers. Without this, on iOS Safari the GPU side
    // doesn't release the buffers until the worker is garbage-collected,
    // which can race with the next worker's WebGPU allocation.
    try { session?.release?.(); } catch (_) {}
    session = null;
    sessionMode = null;
    return { type: 'released', transfer: [] };
  }

  throw new Error(`unknown message type: ${msg.type}`);
}

self.addEventListener('message', async (e) => {
  const msg = e.data;
  try {
    const reply = await handle(msg);
    const transfer = reply.transfer || [];
    delete reply.transfer;
    reply.id = msg.id;
    self.postMessage(reply, transfer);
  } catch (err) {
    self.postMessage({ id: msg.id, type: 'error', message: err.message });
  }
});
