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
let cachedBytes = null;      // Uint8Array of the current model — kept so we
                             // can recreate the session in-place (periodic
                             // refresh) without round-tripping bytes from
                             // the main thread.
let runsSinceRefresh = 0;    // count of session.run() calls on current
                             // session; we refresh every REFRESH_EVERY runs
                             // to bound ORT-internal WebGPU buffer growth.
const REFRESH_EVERY = (() => {
  const ua = (typeof navigator !== 'undefined' && navigator.userAgent) || '';
  // iOS Safari has the tightest WebGPU pool — refresh aggressively. macOS
  // can amortize over many more runs.
  if (/iPhone|iPad|iPod/i.test(ua)) return 4;
  if (/Android/i.test(ua))           return 6;
  return 30;
})();

async function _refreshSession() {
  if (!session || !cachedBytes) return;
  diag(`refreshing ${sessionMode} session after ${runsSinceRefresh} runs`);
  const mode = sessionMode;
  try { session.release(); } catch (_) {}
  session = null;
  const t0 = performance.now();
  session = await withTimeout(
    ort.InferenceSession.create(cachedBytes, { executionProviders: ['webgpu'] }),
    10000, `${mode} webgpu refresh`);
  sessionMode = mode;
  runsSinceRefresh = 0;
  diag(`session refresh done in ${(performance.now()-t0).toFixed(0)}ms`);
}

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
    diag(`init received for mode=${msg.mode} (prior session: ${sessionMode || 'none'})`);
    if (!ort) {
      ort = await import(ORT_BUNDLE);
      if (ort.env && ort.env.wasm) ort.env.wasm.wasmPaths = ORT_BASE;
    }
    const { mode, modelBuf } = msg;
    // Release any prior session WITHIN THIS SAME WORKER before creating the
    // new one. This is the key for iOS: ORT's session.release() actually
    // frees WebGPU device buffers when called inside a live worker context;
    // when the worker is terminated instead, iOS holds the GPU memory
    // indefinitely, starving the next worker's allocation and OOMing.
    if (session) {
      diag(`releasing prior ${sessionMode} session before re-init`);
      try { session.release(); } catch (_) {}
      session = null;
      sessionMode = null;
    }
    const bytes = new Uint8Array(modelBuf);
    const hasWebGPU = await detectWebGPU();
    if (!hasWebGPU) {
      throw new Error('WebGPU not available (this build is WebGPU-only by request — no WASM fallback). On iOS make sure Safari 18+ with WebGPU enabled.');
    }
    const t0 = performance.now();
    session = await withTimeout(
      ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] }),
      10000, `${mode} webgpu init`);
    diag(`WebGPU session ready for ${mode} in ${(performance.now()-t0).toFixed(0)}ms`);
    sessionMode = mode;
    activeBackend = 'webgpu';
    cachedBytes = bytes;          // keep for _refreshSession
    runsSinceRefresh = 0;
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
    let imgT = null, bitsT = null, permT = null, container_rgb = null;
    try {
      imgT  = new ort.Tensor('float32', imgArr,  [1, 3, H, W]);
      bitsT = new ort.Tensor('float32', bitsArr, [1, bitsArr.length]);
      permT = new ort.Tensor('int64',   permI64, [12, 1024, permP]);
      const t0 = performance.now();
      ({ container_rgb } = await session.run({
        host_rgb: imgT, bits: bitsT, perm_stack: permT }));
      runsSinceRefresh++;
      const ms = performance.now() - t0;
      // Copy the Float32 CHW data out of ORT-owned session memory (clamped to
      // [-1, 1] to remove tiny overshoots), then transfer the buffer to the
      // main thread.
      const src = container_rgb.data;
      const f32Out = new Float32Array(src.length);
      for (let i = 0; i < src.length; i++) {
        const v = src[i];
        f32Out[i] = v < -1 ? -1 : (v > 1 ? 1 : v);
      }
      return { type: 'encoded', f32Buf: f32Out.buffer, ms, transfer: [f32Out.buffer] };
    } finally {
      // Dispose unconditionally so a throw inside session.run() doesn't
      // strand the ORT tensor wrappers (which can pin GPU buffers).
      disposeTensor(imgT); disposeTensor(bitsT); disposeTensor(permT);
      disposeTensor(container_rgb);
      if (runsSinceRefresh >= REFRESH_EVERY) await _refreshSession();
    }
  }

  if (msg.type === 'decode') {
    if (sessionMode !== 'decoder') throw new Error('worker not in decoder mode');
    const { imageBuf, W, H, permBuf, permP } = msg;
    const arr = imageBufToFloat32CHW(imageBuf, W, H);
    const permLen = (permBuf.byteLength >>> 2);
    const permI64 = new BigInt64Array(permLen);
    {
      const dv = new DataView(permBuf);
      for (let i = 0; i < permLen; i++) permI64[i] = BigInt(dv.getInt32(i << 2, true));
    }
    let t = null, permT = null, bit_logits = null;
    try {
      t     = new ort.Tensor('float32', arr,     [1, 3, H, W]);
      permT = new ort.Tensor('int64',   permI64, [12, 1024, permP]);
      const t0 = performance.now();
      ({ bit_logits } = await session.run({ container_rgb: t, perm_stack: permT }));
      runsSinceRefresh++;
      const ms = performance.now() - t0;
      const src = bit_logits.data;
      const bits = new Uint8Array(src.length);
      for (let i = 0; i < bits.length; i++) {
        const s = 1 / (1 + Math.exp(-src[i]));
        bits[i] = s > 0.5 ? 1 : 0;
      }
      return { type: 'decoded', bitsBuf: bits.buffer, ms, transfer: [bits.buffer] };
    } finally {
      disposeTensor(t); disposeTensor(permT); disposeTensor(bit_logits);
      if (runsSinceRefresh >= REFRESH_EVERY) await _refreshSession();
    }
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
