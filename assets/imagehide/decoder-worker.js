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
 *   { id, type: 'encode', imageBuf, W, H, bitsBuf }
 *   { id, type: 'decode', imageBuf, W, H }
 *
 * Replies:
 *   { id, type: 'ready'  }
 *   { id, type: 'encoded', imageBuf, ms }
 *   { id, type: 'decoded', bitsBuf,  ms }
 *   { id, type: 'error',   message }
 *
 * All large payloads (imageBuf, bitsBuf) cross the boundary as ArrayBuffer
 * transferables to avoid copies.
 */

let ort = null;
let encoderSession = null;
let decoderSession = null;
let activeBackend = 'wasm';

const ORT_VERSION = '1.20.0';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
// WebGPU bundle includes the WASM provider as fallback.
const ORT_BUNDLE = `${ORT_BASE}ort.webgpu.min.mjs`;

function isSafari() {
  // Safari (and all iOS browsers, which use WebKit) — UA contains "Safari" but
  // not "Chrome" or "CriOS" or "FxiOS". WebKit doesn't reliably support
  // WebGPU in DedicatedWorkers as of 2026: the API may be present but
  // requestAdapter / InferenceSession can hang. Skip WebGPU on Safari.
  const ua = (typeof navigator !== 'undefined' && navigator.userAgent) || '';
  return /Safari/.test(ua) && !/Chrome|Chromium|CriOS|FxiOS|Edg\//.test(ua);
}

async function detectWebGPU() {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) return false;
  if (isSafari()) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch { return false; }
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

function float32CHWtoImageBuf(arr, W, H) {
  const out = new Uint8ClampedArray(H * W * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pi = y * W + x;
      const i = pi * 4;
      out[i]     = Math.round(Math.min(255, Math.max(0, (arr[pi]             + 1) * 127.5)));
      out[i + 1] = Math.round(Math.min(255, Math.max(0, (arr[H * W + pi]     + 1) * 127.5)));
      out[i + 2] = Math.round(Math.min(255, Math.max(0, (arr[2 * H * W + pi] + 1) * 127.5)));
      out[i + 3] = 255;
    }
  }
  return out.buffer;
}

async function handle(msg) {
  if (msg.type === 'init') {
    ort = await import(ORT_BUNDLE);
    if (ort.env && ort.env.wasm) ort.env.wasm.wasmPaths = ORT_BASE;
    // Try WebGPU first, fall back to WASM. Worker termination cleans up both,
    // so the per-decode memory accumulation that previously killed the tab is
    // no longer a constraint here.
    const tryWebGPU = await detectWebGPU();
    const encBytes = new Uint8Array(msg.encoderBuf);
    const decBytes = new Uint8Array(msg.decoderBuf);
    if (tryWebGPU) {
      try {
        // 10s timeout — if WebGPU session create stalls (Safari has been seen
        // to hang here), fall back to WASM rather than hanging the page.
        encoderSession = await withTimeout(
          ort.InferenceSession.create(encBytes, { executionProviders: ['webgpu', 'wasm'] }),
          10000, 'encoder webgpu init');
        decoderSession = await withTimeout(
          ort.InferenceSession.create(decBytes, { executionProviders: ['webgpu', 'wasm'] }),
          10000, 'decoder webgpu init');
        activeBackend = 'webgpu';
        return { type: 'ready', backend: activeBackend, transfer: [] };
      } catch (e) {
        // WebGPU session create failed or timed out — wipe partial state, fall through to WASM.
        try { encoderSession?.release?.(); } catch (_) {}
        try { decoderSession?.release?.(); } catch (_) {}
        encoderSession = null;
        decoderSession = null;
      }
    }
    encoderSession = await ort.InferenceSession.create(encBytes, { executionProviders: ['wasm'] });
    decoderSession = await ort.InferenceSession.create(decBytes, { executionProviders: ['wasm'] });
    activeBackend = 'wasm';
    return { type: 'ready', backend: activeBackend, transfer: [] };
  }

  if (msg.type === 'encode') {
    const { imageBuf, W, H, bitsBuf } = msg;
    const imgArr  = imageBufToFloat32CHW(imageBuf, W, H);
    const bitsArr = new Float32Array(bitsBuf);
    const imgT  = new ort.Tensor('float32', imgArr,  [1, 3, H, W]);
    const bitsT = new ort.Tensor('float32', bitsArr, [1, bitsArr.length]);
    const t0 = performance.now();
    const { container_rgb } = await encoderSession.run({ host_rgb: imgT, bits: bitsT });
    const ms = performance.now() - t0;
    const outBuf = float32CHWtoImageBuf(container_rgb.data, W, H);
    disposeTensor(imgT); disposeTensor(bitsT); disposeTensor(container_rgb);
    return { type: 'encoded', imageBuf: outBuf, ms, transfer: [outBuf] };
  }

  if (msg.type === 'decode') {
    const { imageBuf, W, H } = msg;
    const arr = imageBufToFloat32CHW(imageBuf, W, H);
    const t = new ort.Tensor('float32', arr, [1, 3, H, W]);
    const t0 = performance.now();
    const { bit_logits } = await decoderSession.run({ container_rgb: t });
    const ms = performance.now() - t0;
    const src = bit_logits.data;
    const bits = new Uint8Array(src.length);
    for (let i = 0; i < bits.length; i++) {
      const s = 1 / (1 + Math.exp(-src[i]));
      bits[i] = s > 0.5 ? 1 : 0;
    }
    disposeTensor(t); disposeTensor(bit_logits);
    return { type: 'decoded', bitsBuf: bits.buffer, ms, transfer: [bits.buffer] };
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
