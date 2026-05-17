import { computeCrop, splitTrim, pasteBack } from './trim.js';
import { phash128, packPayload, unpackPayload, bitAccuracy, N_H } from './payload.js';
import { psnr, ssim } from './metrics.js';
import { ATTACKS } from './attacks.js';
import { loadModels, encode, decode, getBackend, releaseSession } from './pipeline.js';

const LIBSODIUM_CDN = 'https://cdn.jsdelivr.net/npm/libsodium-wrappers@0.7.13/+esm';
const IS_MOBILE = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
// Auto-downsample input above this. Mobile is much tighter because WASM-only
// inference is single-threaded and decode time scales ~linearly with pixels.
const MAX_INPUT_PIXELS = IS_MOBILE ? 0.5 * 1024 * 1024 : 2 * 1024 * 1024;
// Attacks pre-checked on first load. Mobile defaults to a smaller representative
// set so the first run completes in tens of seconds, not minutes.
const DEFAULT_CHECKED = IS_MOBILE
  ? new Set(['identity', 'jpeg_q80', 'jpeg_q40', 'chain_insta', 'chain_wechat'])
  : null;   // null = check everything

const els = {};
let sodium = null;
let demoKeypair = null;
let state = {
  status: 'idle',                 // idle | loading-model | ready | encoding | running-attacks | error
  originalImage: null,            // full ImageData
  crop: null,                     // computeCrop result
  coreCover: null,                // cropped cover ImageData
  fullCover: null,                // full reconstructed cover (for paste-back)
  fullContainer: null,            // full-size container after encode (for download)
  coreContainer: null,            // cropped container
  payloadBits: null,              // Uint8Array(896)
  payloadParts: null,             // { H, sig, pk }
  encodeMs: null,
  decodeMs: null,
  sizeOk: false,
};

document.addEventListener('DOMContentLoaded', init);

function init() {
  ['upload', 'file', 'sample', 'status', 'cover', 'container', 'residual',
   'oneshot', 'attackList', 'runBtn', 'resultsBody'].forEach(k => {
    els[k] = document.getElementById(`ih-${k}`);
  });
  els.file.addEventListener('change', e => onFile(e.target.files?.[0]));
  els.upload.addEventListener('dragover', e => {
    e.preventDefault(); els.upload.classList.add('is-dragover');
  });
  els.upload.addEventListener('dragleave', () => els.upload.classList.remove('is-dragover'));
  els.upload.addEventListener('drop', e => {
    e.preventDefault();
    els.upload.classList.remove('is-dragover');
    onFile(e.dataTransfer.files?.[0]);
  });
  els.sample.addEventListener('click', e => { e.preventDefault(); loadSample(); });
  els.runBtn.addEventListener('click', onRun);
  renderAttackList();
  setStatus('Drop an image or pick a file to begin.');
}

function renderAttackList() {
  const isChecked = a => !DEFAULT_CHECKED || DEFAULT_CHECKED.has(a.id);
  els.attackList.innerHTML = ATTACKS.map(a =>
    `<label><input type="checkbox" value="${a.id}"${isChecked(a) ? ' checked' : ''}> ${a.label}</label>`
  ).join('');
  els.resultsBody.innerHTML = ATTACKS.map(a =>
    `<tr id="row-${a.id}" class="queued">
       <td>${a.label}</td><td class="num">—</td><td class="num">—</td>
       <td class="num">—</td><td>—</td>
     </tr>`).join('');
}

function setStatus(html, isError = false) {
  els.status.innerHTML = html;
  els.status.classList.toggle('warn', isError);
}

function bitmapToFittedImageData(bitmap) {
  // Downsample (preserve aspect ratio, no crop) so total pixels <= MAX_INPUT_PIXELS.
  // Returns { imageData, origW, origH }.
  const origW = bitmap.width, origH = bitmap.height;
  let w = origW, h = origH;
  const pixels = w * h;
  if (pixels > MAX_INPUT_PIXELS) {
    const s = Math.sqrt(MAX_INPUT_PIXELS / pixels);
    w = Math.max(64, Math.round(w * s));
    h = Math.max(64, Math.round(h * s));
  }
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'medium';
  ctx.drawImage(bitmap, 0, 0, w, h);
  const imageData = ctx.getImageData(0, 0, w, h);
  c.width = 0; c.height = 0;
  bitmap.close();
  return { imageData, origW, origH };
}

async function onFile(file) {
  if (!file) return;
  setStatus(`Loading ${file.name}…`);
  try {
    const bitmap = await createImageBitmap(file);
    const { imageData, origW, origH } = bitmapToFittedImageData(bitmap);
    onImageLoaded(imageData, origW, origH);
  } catch (e) {
    setStatus(`Failed to load: ${e.message}`, true);
  }
}

async function loadSample() {
  setStatus('Loading sample…');
  try {
    const resp = await fetch('/assets/imagehide/sample-cover.jpg');
    const blob = await resp.blob();
    const bitmap = await createImageBitmap(blob);
    const { imageData, origW, origH } = bitmapToFittedImageData(bitmap);
    onImageLoaded(imageData, origW, origH);
  } catch (e) {
    setStatus(`Failed to load sample: ${e.message}`, true);
  }
}

function onImageLoaded(imageData, origW, origH) {
  state.originalImage = imageData;
  const W = imageData.width, H = imageData.height;
  state.crop = computeCrop(H, W);
  drawToCanvas(els.cover, imageData);
  const downsampleNote = (origW && origH && (W !== origW || H !== origH))
    ? ` (downsampled from ${origW}×${origH} — ${(origW * origH / 1e6).toFixed(1)} MP — to fit browser memory; aspect preserved, no crop)`
    : '';
  setStatus(`Loaded ${W}×${H}${downsampleNote}. Encoding ${state.crop.cropW}×${state.crop.cropH} (multiple of 64).`);
  ensureLoaded();
}

function drawToCanvas(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext('2d').putImageData(
    new ImageData(imageData.data, imageData.width, imageData.height), 0, 0);
}

async function ensureLoaded() {
  if (state.status === 'ready') {
    els.runBtn.disabled = false;
    return;
  }
  if (state.status === 'loading-model') return;
  state.status = 'loading-model';
  setStatus('Loading model… 0%');
  try {
    await Promise.all([loadSodium(), loadModel()]);
    state.status = 'ready';
    els.runBtn.disabled = false;
    const backendNote = getBackend() === 'wasm'
      ? ' <span class="warn">Running on WASM (no WebGPU) — expect 5–10× slower.</span>'
      : '';
    setStatus(`Model loaded (backend: ${getBackend()}). Click Run to encode + attack.${backendNote}`);
  } catch (e) {
    state.status = 'error';
    setStatus(`Model load failed: ${e.message}`, true);
  }
}

async function loadSodium() {
  if (sodium) return;
  const mod = await import(/* @vite-ignore */ LIBSODIUM_CDN);
  sodium = mod.default || mod;
  await sodium.ready;
  demoKeypair = sodium.crypto_sign_keypair();
}

let loaded = { encoder: 0, decoder: 0, total: { encoder: 0, decoder: 0 } };
async function loadModel() {
  await loadModels(
    '/assets/imagehide/encoder.onnx',
    '/assets/imagehide/decoder.onnx',
    ({ tag, loaded: l, total }) => {
      loaded[tag] = l;
      if (total) loaded.total[tag] = total;
      const sum = loaded.encoder + loaded.decoder;
      const tot = loaded.total.encoder + loaded.total.decoder;
      const pct = tot ? Math.round(100 * sum / tot) : 0;
      setStatus(`Loading model… <span class="progress"><span style="width:${pct}%"></span></span> ${pct}% (${(sum/1e6).toFixed(1)} / ${(tot/1e6).toFixed(1)} MB)`);
    },
  );
}

async function onRun() {
  if (state.status !== 'ready' && state.status !== 'done') return;
  els.runBtn.disabled = true;
  state.status = 'encoding';
  setStatus('Encoding…');
  try {
    await runEncode();
    await runAttacks();
    state.status = 'done';
    els.runBtn.disabled = false;
  } catch (e) {
    state.status = 'error';
    setStatus(`Run failed: ${e.message}`, true);
    els.runBtn.disabled = false;
  }
}

async function runEncode() {
  const { core, strips } = splitTrim(state.originalImage, state.crop);
  state.coreCover = core;
  // Fully reconstructed cover = original (paste-back over same strips is a no-op
  // for the cover). Keep a reference for later attack pairs.
  state.fullCover = state.originalImage;

  // Build payload: H = pHash(core), sig = Ed25519(H), pk = demo public.
  const H_bytes = phash128(core);
  const sig = sodium.crypto_sign_detached(H_bytes, demoKeypair.privateKey);
  const pk = demoKeypair.publicKey;
  const bits = packPayload(H_bytes, sig, pk);
  state.payloadBits = bits;
  state.payloadParts = { H: H_bytes, sig, pk };

  const { container, ms } = await encode(core, bits);
  state.coreContainer = container;
  state.encodeMs = ms;

  // Reconstruct full-size container (untouched strips + encoded core).
  const fullContainer = pasteBack(container, strips, state.crop,
                                  state.originalImage.width,
                                  state.originalImage.height);
  drawToCanvas(els.container, fullContainer);
  drawResidual(els.residual, core, container, 10);

  // Capture original dimensions for the oneshot line, then drop the full-size
  // buffers — attacks only need (capped, ≤1024²) coreCover/coreContainer from
  // here on, which is critical for mobile memory budgets.
  state.origW = state.originalImage.width;
  state.origH = state.originalImage.height;
  state.fullContainer = null;
  state.fullCover = null;
  state.originalImage = null;

  // Single decode time-probe on the clean container.
  const dec = await decode(container);
  state.decodeMs = dec.ms;

  const cropMsg = (state.crop.trimmedTop || state.crop.trimmedLeft)
    ? ` (${state.crop.trimmedTop + state.crop.trimmedBottom} px trimmed vertically, ${state.crop.trimmedLeft + state.crop.trimmedRight} px horizontally)`
    : '';
  els.oneshot.textContent =
    `Image: ${state.origW} × ${state.origH} → encoded region ${state.crop.cropW} × ${state.crop.cropH}${cropMsg}\n` +
    `Payload: 896 bits (128 H | 512 sig | 256 pk)\n` +
    `Encode: ${state.encodeMs.toFixed(1)} ms · Decode: ${state.decodeMs.toFixed(1)} ms`;
}

function drawResidual(canvas, coverCore, containerCore, scale) {
  const W = coverCore.width, H = coverCore.height;
  const data = new Uint8ClampedArray(H * W * 4);
  for (let i = 0; i < data.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const d = containerCore.data[i + c] - coverCore.data[i + c];
      data[i + c] = Math.max(0, Math.min(255, 128 + d * scale));
    }
    data[i + 3] = 255;
  }
  canvas.width = W; canvas.height = H;
  canvas.getContext('2d').putImageData(new ImageData(data, W, H), 0, 0);
}

async function runAttacks() {
  state.status = 'running-attacks';
  const enabled = new Set(
    [...els.attackList.querySelectorAll('input[type="checkbox"]:checked')]
      .map(el => el.value));
  // Reset rows
  for (const a of ATTACKS) {
    const row = document.getElementById(`row-${a.id}`);
    row.classList.toggle('queued', enabled.has(a.id));
    row.classList.toggle('running', false);
    row.querySelectorAll('td').forEach((td, idx) => { if (idx > 0) td.textContent = '—'; td.classList.remove('sig-ok', 'sig-no'); });
  }
  for (const a of ATTACKS) {
    if (!enabled.has(a.id)) continue;
    const row = document.getElementById(`row-${a.id}`);
    row.classList.remove('queued');
    row.classList.add('running');
    // Long enough yield for the browser to GC, paint, and release GPU buffers
    // from the prior decode. Critical on mobile Safari where memory pressure
    // accumulates faster than rAF-only yields let it recover.
    await new Promise(r => setTimeout(r, 150));
    try {
      // Attacks run on the (already-cropped, capped-size) core, not the full
      // original — saves an order of magnitude of memory on phone-sized inputs.
      const aContCore = await a.fn(state.coreContainer);
      const aCovCore  = await a.fn(state.coreCover);
      const p = psnr(aContCore, aCovCore);
      const s = ssim(aContCore, aCovCore);
      const { bits: recBits } = await decode(aContCore);
      const acc = bitAccuracy(recBits, state.payloadBits);
      const { H: recH, sig: recSig } = unpackPayload(recBits);
      const sigOk = sodium.crypto_sign_verify_detached(recSig, recH, demoKeypair.publicKey);
      const tds = row.querySelectorAll('td');
      tds[1].textContent = isFinite(p) ? `${p.toFixed(1)} dB` : '∞ dB';
      tds[2].textContent = s.toFixed(3);
      tds[3].textContent = acc.toFixed(3);
      tds[4].textContent = sigOk ? '✓' : '✗';
      tds[4].classList.add(sigOk ? 'sig-ok' : 'sig-no');
    } catch (e) {
      const tds = row.querySelectorAll('td');
      tds[1].textContent = 'err';
      tds[1].title = e.message;
      setStatus(`Attack ${a.label} failed: ${e.message}`, true);
    }
    row.classList.remove('running');
  }
  // Terminate the inference worker so its WASM heap is released to the OS.
  // The next Run will re-spawn it cheaply from the cached model bytes.
  releaseSession();
  setStatus('Done. (worker terminated; WASM memory freed)');
}
