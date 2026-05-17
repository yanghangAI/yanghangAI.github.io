/**
 * Worker-backed inference pipeline.
 *
 * All ONNX inference runs in a Web Worker so we can `worker.terminate()`
 * to reclaim WASM linear memory — the only browser mechanism that actually
 * shrinks WASM heap (it doesn't have a free/shrink API). The model bytes are
 * cached in main-thread memory so re-spawning the worker is just session-
 * create, not a network fetch.
 *
 * Memory strategy: the worker hosts ONE session at a time — either the
 * encoder OR the decoder. Switching modes terminates the worker and re-spawns
 * it with only the needed model loaded. This roughly halves resident WASM
 * footprint vs. loading both sessions into the same worker.
 *
 * Public API:
 *   getBackend()                  → 'wasm' | 'webgpu'
 *   loadModels(encoderUrl, decoderUrl, onProgress)
 *   encode(coverImageData, bitsUint8) → { container, ms }
 *   decode(containerImageData)        → { bits, ms }
 *   releaseSession()              → terminates worker; bytes kept cached
 */

let worker = null;
let workerMode = null;        // null | 'encoder' | 'decoder'
let encoderBuf = null;
let decoderBuf = null;
let activeBackend = 'wasm';   // worker uses WASM; reported to UI for honesty
let nextMsgId = 1;
const pending = new Map();

// Worker URL with cache-bust query so a redeploy reloads the worker source.
const V = (typeof self !== 'undefined' && self.__imagehideVersion) ||
          (typeof window !== 'undefined' && window.__imagehideVersion) || 'dev';
const WORKER_URL = new URL(`./decoder-worker.js?v=${V}`, import.meta.url);

function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(WORKER_URL, { type: 'module' });
  worker.addEventListener('message', e => {
    const msg = e.data;
    const p = pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    if (msg.type === 'error') p.reject(new Error(msg.message));
    else p.resolve(msg);
  });
  worker.addEventListener('error', e => {
    // Surface uncaught worker errors to all pending promises
    const err = new Error(`worker error: ${e.message || 'unknown'}`);
    for (const p of pending.values()) p.reject(err);
    pending.clear();
  });
  return worker;
}

function send(msg, transfer = []) {
  const w = ensureWorker();
  msg.id = nextMsgId++;
  return new Promise((resolve, reject) => {
    pending.set(msg.id, { resolve, reject });
    w.postMessage(msg, transfer);
  });
}

export function getBackend() { return activeBackend; }

export async function loadModels(encoderUrl, decoderUrl, onProgress) {
  if (!encoderBuf || !decoderBuf) {
    const [encBuf, decBuf] = await Promise.all([
      fetchWithProgress(encoderUrl, onProgress, 'encoder'),
      fetchWithProgress(decoderUrl, onProgress, 'decoder'),
    ]);
    encoderBuf = encBuf;
    decoderBuf = decBuf;
  }
  // No worker spawn here — encode()/decode() lazily start a single-mode worker.
}

async function ensureMode(mode) {
  if (workerMode === mode && worker) return;
  // Switching modes (or first use): tear down any existing worker so the
  // WASM heap of the prior session is fully released before we load the next.
  if (worker) {
    worker.terminate();
    worker = null;
  }
  workerMode = null;
  ensureWorker();
  const buf = mode === 'encoder' ? encoderBuf : decoderBuf;
  const clone = buf.slice(0);
  const reply = await send(
    { type: 'init', mode, modelBuf: clone },
    [clone],
  );
  if (reply.backend) activeBackend = reply.backend;
  workerMode = mode;
}

async function fetchWithProgress(url, cb, tag) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url} → ${resp.status}`);
  const total = Number(resp.headers.get('content-length')) || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (cb) cb({ tag, loaded, total });
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf.buffer;
}

export async function encode(coverImageData, bitsUint8) {
  await ensureMode('encoder');
  const W = coverImageData.width, H = coverImageData.height;
  // We must transfer; the caller can no longer use these buffers. We slice to
  // copy so the caller's ImageData stays usable.
  const imageBuf = coverImageData.data.buffer.slice(0);
  // Convert 0/1 bytes → Float32 values up front. Sending the raw byte buffer
  // and wrapping it as Float32 on the worker side would reinterpret 4 bytes
  // per float, yielding 224 garbage floats instead of 896 valid ones.
  const bitsFloat = new Float32Array(bitsUint8.length);
  for (let i = 0; i < bitsUint8.length; i++) bitsFloat[i] = bitsUint8[i];
  const bitsBuf = bitsFloat.buffer;
  const r = await send(
    { type: 'encode', imageBuf, W, H, bitsBuf },
    [imageBuf, bitsBuf],
  );
  // Worker now returns Float32 CHW directly. We keep the float32 buffer for the
  // attack pipeline (sub-uint8 residual survives the resize step) and also
  // materialize a uint8 ImageData copy for display / pasteBack / PSNR.
  const containerF32 = new Float32Array(r.f32Buf);
  const u8 = new Uint8ClampedArray(H * W * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pi = y * W + x;
      const i  = pi * 4;
      u8[i]     = Math.round(Math.max(0, Math.min(255, (containerF32[pi]             + 1) * 127.5)));
      u8[i + 1] = Math.round(Math.max(0, Math.min(255, (containerF32[H * W + pi]     + 1) * 127.5)));
      u8[i + 2] = Math.round(Math.max(0, Math.min(255, (containerF32[2 * H * W + pi] + 1) * 127.5)));
      u8[i + 3] = 255;
    }
  }
  return {
    container: { data: u8, width: W, height: H },
    containerF32,  // Float32Array, length 3*H*W, CHW, [-1, 1]
    width: W,
    height: H,
    ms: r.ms,
  };
}

export async function decode(containerImageData) {
  await ensureMode('decoder');
  const W = containerImageData.width, H = containerImageData.height;
  const imageBuf = containerImageData.data.buffer.slice(0);
  const r = await send(
    { type: 'decode', imageBuf, W, H },
    [imageBuf],
  );
  return { bits: new Uint8Array(r.bitsBuf), ms: r.ms };
}

/**
 * Terminate the worker, releasing its WASM heap back to the OS. The cached
 * model bytes are kept in main memory so the next `encode`/`decode` call will
 * lazily re-spawn the worker via `initWorker()`.
 */
export function releaseSession() {
  if (worker) {
    worker.terminate();
    worker = null;
  }
  workerMode = null;
  for (const p of pending.values()) p.reject(new Error('session released'));
  pending.clear();
}
