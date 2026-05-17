/**
 * ONNX Runtime Web wrapper for the MWIP encoder/decoder graphs.
 *
 * Encoder input:  host_rgb tensor [1, 3, H, W] float32 in [-1, 1]
 *                 bits     tensor [1, 896]     float32 in {0, 1}
 * Encoder output: container_rgb [1, 3, H, W] float32 in [-1, 1]
 *
 * Decoder input:  container_rgb [1, 3, H, W] float32 in [-1, 1]
 * Decoder output: bit_logits   [1, 896]    float32 (apply sigmoid + 0.5 threshold)
 *
 * Backend: WebGPU when available, falls back to WASM. Surface the active
 * backend via getBackend() so the UI can warn users on the slow path.
 */

let ort = null;          // loaded lazily
let encoderSession = null;
let decoderSession = null;
let activeBackend = null;
let loadPromise = null;

const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.webgpu.min.mjs';

function isMobile() {
  return /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
}

async function detectWebGPU() {
  if (!('gpu' in navigator)) return false;
  // iOS Safari WebGPU sessions accumulate compute-pipeline memory across
  // inference calls and kill tabs shortly after a batch completes. WASM is
  // slower but memory-stable. Same observed pattern on Android Chrome under
  // ORT-web 1.20 — gate by UA to be safe.
  if (isMobile()) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch { return false; }
}

export function getBackend() { return activeBackend; }

export async function loadModels(encoderUrl, decoderUrl, onProgress) {
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    if (!ort) ort = await import(/* @vite-ignore */ ORT_CDN);
    activeBackend = (await detectWebGPU()) ? 'webgpu' : 'wasm';
    // Fetch both .onnx files with a shared progress callback.
    const [encBuf, decBuf] = await Promise.all([
      fetchWithProgress(encoderUrl, onProgress, 'encoder'),
      fetchWithProgress(decoderUrl, onProgress, 'decoder'),
    ]);
    const sessionOpts = { executionProviders: [activeBackend] };
    encoderSession = await ort.InferenceSession.create(encBuf, sessionOpts);
    decoderSession = await ort.InferenceSession.create(decBuf, sessionOpts);
  })();
  return loadPromise;
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

// ---------- tensor helpers ----------

function imageDataToFloat32CHW(imageData) {
  // RGBA [0..255] → CHW [-1..1] float32
  const { data, width: W, height: H } = imageData;
  const out = new Float32Array(3 * H * W);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const pi = y * W + x;
      out[0 * H * W + pi] = (data[i]     / 127.5) - 1;
      out[1 * H * W + pi] = (data[i + 1] / 127.5) - 1;
      out[2 * H * W + pi] = (data[i + 2] / 127.5) - 1;
    }
  }
  return out;
}

function float32CHWtoImageData(arr, W, H) {
  const data = new Uint8ClampedArray(H * W * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pi = y * W + x;
      const i = pi * 4;
      data[i]     = Math.round(Math.min(255, Math.max(0, (arr[0 * H * W + pi] + 1) * 127.5)));
      data[i + 1] = Math.round(Math.min(255, Math.max(0, (arr[1 * H * W + pi] + 1) * 127.5)));
      data[i + 2] = Math.round(Math.min(255, Math.max(0, (arr[2 * H * W + pi] + 1) * 127.5)));
      data[i + 3] = 255;
    }
  }
  return { data, width: W, height: H };
}

function bitsToFloat32(bitsUint8) {
  const out = new Float32Array(bitsUint8.length);
  for (let i = 0; i < bitsUint8.length; i++) out[i] = bitsUint8[i];
  return out;
}

// ---------- public API ----------

function disposeTensor(t) {
  // WebGPU tensors hold GPU buffers that JS GC cannot free; CPU/WASM tensors
  // gain nothing from dispose() but it's safe to call.
  if (t && typeof t.dispose === 'function') {
    try { t.dispose(); } catch (_) { /* ignore */ }
  }
}

export async function encode(coverImageData, bitsUint8) {
  if (!encoderSession) throw new Error('encoder not loaded');
  const W = coverImageData.width, H = coverImageData.height;
  const imgArr = imageDataToFloat32CHW(coverImageData);
  const imgT = new ort.Tensor('float32', imgArr, [1, 3, H, W]);
  const bitsT = new ort.Tensor('float32', bitsToFloat32(bitsUint8), [1, bitsUint8.length]);
  const t0 = performance.now();
  const { container_rgb } = await encoderSession.run({ host_rgb: imgT, bits: bitsT });
  const dt = performance.now() - t0;
  const container = float32CHWtoImageData(container_rgb.data, W, H);
  disposeTensor(imgT);
  disposeTensor(bitsT);
  disposeTensor(container_rgb);
  return { container, ms: dt };
}

export async function decode(containerImageData) {
  if (!decoderSession) throw new Error('decoder not loaded');
  const W = containerImageData.width, H = containerImageData.height;
  const arr = imageDataToFloat32CHW(containerImageData);
  const t = new ort.Tensor('float32', arr, [1, 3, H, W]);
  const t0 = performance.now();
  const { bit_logits } = await decoderSession.run({ container_rgb: t });
  const dt = performance.now() - t0;
  const src = bit_logits.data;
  const bits = new Uint8Array(src.length);
  for (let i = 0; i < bits.length; i++) {
    const s = 1 / (1 + Math.exp(-src[i]));
    bits[i] = s > 0.5 ? 1 : 0;
  }
  disposeTensor(t);
  disposeTensor(bit_logits);
  return { bits, ms: dt };
}
