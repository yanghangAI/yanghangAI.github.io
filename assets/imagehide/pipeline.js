/**
 * Worker-backed inference pipeline.
 *
 * All ONNX inference runs in a Web Worker so we can `worker.terminate()`
 * between attack batches and reclaim WASM linear memory — the only browser
 * mechanism that actually shrinks WASM heap (it doesn't have a free/shrink
 * API). The model bytes are cached in main-thread memory so re-spawning the
 * worker is just session-create, not a network fetch.
 *
 * Public API:
 *   getBackend()                  → 'wasm' | 'webgpu'
 *   loadModels(encoderUrl, decoderUrl, onProgress)
 *   encode(coverImageData, bitsUint8) → { container, ms }
 *   decode(containerImageData)        → { bits, ms }
 *   releaseSession()              → terminates worker; bytes kept cached
 */

let worker = null;
let workerReady = false;
let encoderBuf = null;
let decoderBuf = null;
let activeBackend = 'wasm';   // worker uses WASM; reported to UI for honesty
let nextMsgId = 1;
const pending = new Map();

const WORKER_URL = new URL('./decoder-worker.js', import.meta.url);

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
  await initWorker();
}

async function initWorker() {
  if (workerReady) return;
  ensureWorker();
  // Send clones (slice copies the buffer) and transfer the clones so the
  // worker takes ownership; we keep the originals for the next batch.
  const encClone = encoderBuf.slice(0);
  const decClone = decoderBuf.slice(0);
  await send(
    { type: 'init', encoderBuf: encClone, decoderBuf: decClone },
    [encClone, decClone],
  );
  workerReady = true;
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
  if (!workerReady) await initWorker();
  const W = coverImageData.width, H = coverImageData.height;
  // We must transfer; the caller can no longer use these buffers. We slice to
  // copy so the caller's ImageData stays usable.
  const imageBuf = coverImageData.data.buffer.slice(0);
  const bitsBuf  = bitsUint8.buffer.slice(0);
  const r = await send(
    { type: 'encode', imageBuf, W, H, bitsBuf },
    [imageBuf, bitsBuf],
  );
  return {
    container: { data: new Uint8ClampedArray(r.imageBuf), width: W, height: H },
    ms: r.ms,
  };
}

export async function decode(containerImageData) {
  if (!workerReady) await initWorker();
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
  workerReady = false;
  // Reject any pending operations
  for (const p of pending.values()) p.reject(new Error('session released'));
  pending.clear();
}
