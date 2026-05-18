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
// Once a WebGPU session creation has failed in this page-session, remember it
// and skip the WebGPU attempt on every subsequent worker spawn. Otherwise on
// mobile we pay the WebGPU init cost (and risk GPU-mem fragmentation that
// OOMs the WASM fallback) on every mode switch.
let _webgpuKnownBad = false;
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
    // Out-of-band diagnostics from the worker (id=0). Surface them in the
    // main-thread console with a clear prefix since browsers like Safari
    // hide worker console output by default.
    if (msg && msg.type === 'diag') {
      console.log('[imagehide-worker]', msg.message);
      return;
    }
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

export async function loadModels(encoderUrl, decoderUrl, onProgress,
                                 { eagerInit = true } = {}) {
  if (!encoderBuf || !decoderBuf) {
    const [encBuf, decBuf] = await Promise.all([
      fetchWithProgress(encoderUrl, onProgress, 'encoder'),
      fetchWithProgress(decoderUrl, onProgress, 'decoder'),
    ]);
    encoderBuf = encBuf;
    decoderBuf = decBuf;
  }
  // Optionally spawn an encoder-mode worker now so the WebGPU/WASM negotiation
  // happens before the user clicks anything (and getBackend() reports the
  // real backend in the status bar). On mobile we skip this — eagerly
  // holding ~100 MB of WASM heap while the user is idle reading the page
  // pushes iOS Safari into OOM territory. Mobile pays a one-time ~500 ms-1s
  // session-init cost on the first encode click instead.
  if (eagerInit) await ensureMode('encoder');
}

function _isMobileUA() {
  const ua = (typeof navigator !== 'undefined' && navigator.userAgent) || '';
  return /iPhone|iPad|iPod|Android/i.test(ua);
}

function _sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function _gracefulShutdownWorker() {
  if (!worker) return;
  // 1. Ask the worker to release its ORT session cleanly. This flushes any
  //    in-flight WebGPU command buffers so the GPU side actually drops the
  //    device-backed tensors instead of waiting for the dead worker's GC.
  //    Best-effort: cap at 2 s so a wedged worker can't hang the page.
  try {
    await Promise.race([
      send({ type: 'release' }),
      _sleep(2000),
    ]);
  } catch (_) { /* worker may already be wedged */ }
  // 2. Now actually terminate the worker thread.
  worker.terminate();
  worker = null;
  workerMode = null;
  // Clear any pending callbacks left over from the in-flight release msg.
  for (const p of pending.values()) p.reject(new Error('session released'));
  pending.clear();
  // 3. On iOS/Android the WebGPU device + WASM heap reclaim is asynchronous.
  //    Pause briefly so the next worker's allocation doesn't race against
  //    not-yet-released GPU buffers from the previous worker.
  if (_isMobileUA()) await _sleep(400);
}

async function _spawnAndInit(mode, opts) {
  await _gracefulShutdownWorker();
  ensureWorker();
  const buf = mode === 'encoder' ? encoderBuf : decoderBuf;
  const clone = buf.slice(0);
  // If WebGPU has already failed once this session, don't even attempt it
  // again — the repeated GPU-mem allocate/release thrash is what's been
  // OOMing iPhone Safari on the decode side.
  const initOpts = { ...(opts || {}) };
  if (_webgpuKnownBad) initOpts.forceWasm = true;
  const reply = await send(
    { type: 'init', mode, modelBuf: clone, ...initOpts },
    [clone],
  );
  if (reply.backend) activeBackend = reply.backend;
  workerMode = mode;
}

async function ensureMode(mode) {
  if (workerMode === mode && worker) return;
  try {
    await _spawnAndInit(mode);
  } catch (e) {
    // ORT-Web's WASM init can get marked as permanently failed when the
    // ['webgpu', 'wasm'] provider list crashes halfway — typical on iOS
    // Safari, where WebGPU may fail to create a session for some op and
    // leave WASM init in a "previous call to initWasm() failed" state.
    // The failure is cached on the ORT instance, which lives inside the
    // worker, so respawning the worker and retrying with WebGPU disabled
    // gives us a clean ORT instance with no poisoned cache.
    const m = e && e.message ? e.message : '';
    if (/initWasm|no available backend|previous call|Out of memory/i.test(m)) {
      _webgpuKnownBad = true;  // suppress further WebGPU attempts this session
      console.warn('[imagehide] worker init failed, respawning WASM-only (and disabling WebGPU for the rest of this session):', m);
      await _spawnAndInit(mode, { forceWasm: true });
    } else {
      throw e;
    }
  }
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

export async function encode(coverImageData, bitsUint8, perm) {
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
  // perm: { data: Int32Array, p } — slice to copy then transfer ownership.
  const permBuf = perm.data.buffer.slice(0);
  const permP = perm.p;
  const r = await send(
    { type: 'encode', imageBuf, W, H, bitsBuf, permBuf, permP },
    [imageBuf, bitsBuf, permBuf],
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

export async function decode(containerImageData, perm) {
  await ensureMode('decoder');
  const W = containerImageData.width, H = containerImageData.height;
  const imageBuf = containerImageData.data.buffer.slice(0);
  const permBuf = perm.data.buffer.slice(0);
  const permP = perm.p;
  const r = await send(
    { type: 'decode', imageBuf, W, H, permBuf, permP },
    [imageBuf, permBuf],
  );
  return { bits: new Uint8Array(r.bitsBuf), ms: r.ms };
}

/**
 * Gracefully release the worker: ask ORT to release its session first
 * (flushes WebGPU/WASM device resources), then terminate the worker so
 * the WASM linear memory is returned to the OS. Cached model bytes are
 * kept in main memory so the next `encode`/`decode` call lazily re-spawns.
 *
 * Returns a Promise that resolves once the shutdown is fully complete —
 * call sites that need memory freed before continuing should await it.
 */
export async function releaseSession() {
  await _gracefulShutdownWorker();
}
