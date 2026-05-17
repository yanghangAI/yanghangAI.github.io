// Versioned dynamic imports — the HTML stamps `window.__imagehideVersion`
// from Jekyll build time so a redeploy busts every cached sibling module.
const V = (typeof window !== 'undefined' && window.__imagehideVersion) || 'dev';
const { computeCrop, splitTrim, pasteBack } = await import(`./trim.js?v=${V}`);
const { phash128, packPayload, unpackPayload, bitAccuracy, bitsToBytes,
        N_H, N_SIG, N_BITS } = await import(`./payload.js?v=${V}`);
const { psnr, ssim } = await import(`./metrics.js?v=${V}`);
const { ATTACKS } = await import(`./attacks.js?v=${V}`);
const { loadModels, encode, decode, getBackend, releaseSession } =
  await import(`./pipeline.js?v=${V}`);

const LIBSODIUM_CDN = 'https://cdn.jsdelivr.net/npm/libsodium-wrappers@0.7.13/+esm';
const IS_MOBILE = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
const MAX_INPUT_PIXELS = IS_MOBILE ? 0.5 * 1024 * 1024 : 1 * 1024 * 1024;

// ---------- shared state ----------
const els = {};
let sodium = null;
let demoKeypair = null;
let modelStatus = 'idle';   // 'idle' | 'loading' | 'ready' | 'error'

// Across-mode handoff: the last encoded container ImageData + its payload bits
// (so decode mode can verify bit accuracy without re-encoding).
let lastContainer = null;       // ImageData
let lastPayloadBits = null;     // Uint8Array(896)

// ---------- encode mode state ----------
const enc = {
  coverImage: null,    // ImageData (downsampled cover)
  origW: 0, origH: 0,
  crop: null,
  busy: false,
};

// ---------- decode mode state ----------
const dec = {
  uploadImage: null,   // ImageData (downsampled upload)
  busy: false,
};

// ---------- init ----------
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

function init() {
  [
    // encode pane
    'enc-upload', 'enc-file', 'enc-sample', 'enc-customBits', 'enc-runBtn',
    'enc-downloadBtn', 'enc-status', 'cover', 'container', 'residual', 'oneshot',
    // decode pane
    'dec-srcLast', 'dec-srcUpload', 'dec-lastHint', 'dec-upload', 'dec-file',
    'dec-attack', 'dec-runBtn', 'dec-status', 'dec-preview', 'dec-previewLabel',
    'dec-output',
  ].forEach(k => { els[k] = document.getElementById(`ih-${k}`); });

  initTabs();
  initEncodePane();
  initDecodePane();
  loadInfra();
}

// ---------- tabs ----------
function initTabs() {
  document.querySelectorAll('.imagehide__tab').forEach(btn => {
    btn.addEventListener('click', () => switchMode(btn.dataset.mode));
  });
}

function switchMode(mode) {
  document.querySelectorAll('.imagehide__tab').forEach(b => {
    const on = b.dataset.mode === mode;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-selected', on ? 'true' : 'false');
  });
  document.querySelectorAll('.imagehide__pane').forEach(p => {
    p.classList.toggle('is-hidden', p.dataset.pane !== mode);
  });
  // Tearing down any worker the prior mode left running frees memory before
  // the other mode spins up its own session.
  releaseSession();
}

// ---------- infra: model + libsodium ----------
async function loadInfra() {
  if (modelStatus !== 'idle') return;
  modelStatus = 'loading';
  setBothStatus('Loading model… 0%');
  try {
    await Promise.all([loadSodium(), loadModel()]);
    modelStatus = 'ready';
    const backendNote = ' (backend: ' + getBackend() + ')';
    setEncStatus('Pick a cover image to encode' + backendNote);
    setDecStatus('Encode an image, or upload one to decode' + backendNote);
    refreshEncRunBtn();
    refreshDecRunBtn();
  } catch (e) {
    modelStatus = 'error';
    setBothStatus(`Model load failed: ${e.message}`, true);
  }
}

function setBothStatus(msg, isErr) {
  setEncStatus(msg, isErr);
  setDecStatus(msg, isErr);
}

async function loadSodium() {
  if (sodium) return;
  const mod = await import(/* @vite-ignore */ LIBSODIUM_CDN);
  sodium = mod.default || mod;
  await sodium.ready;
  demoKeypair = sodium.crypto_sign_keypair();
}

const modelProgress = { encoder: 0, decoder: 0, total: { encoder: 0, decoder: 0 } };
async function loadModel() {
  await loadModels(
    '/assets/imagehide/encoder.onnx',
    '/assets/imagehide/decoder.onnx',
    ({ tag, loaded, total }) => {
      modelProgress[tag] = loaded;
      if (total) modelProgress.total[tag] = total;
      const sum = modelProgress.encoder + modelProgress.decoder;
      const tot = modelProgress.total.encoder + modelProgress.total.decoder;
      const pct = tot ? Math.round(100 * sum / tot) : 0;
      const html = `Loading model… <span class="progress"><span style="width:${pct}%"></span></span> ${pct}% (${(sum/1e6).toFixed(1)} / ${(tot/1e6).toFixed(1)} MB)`;
      setBothStatus(html);
    },
  );
}

// ---------- helpers ----------
function setEncStatus(html, isErr = false) {
  els['enc-status'].innerHTML = html;
  els['enc-status'].classList.toggle('warn', isErr);
}
function setDecStatus(html, isErr = false) {
  els['dec-status'].innerHTML = html;
  els['dec-status'].classList.toggle('warn', isErr);
}

function bitmapToFitted(bitmap) {
  const origW = bitmap.width, origH = bitmap.height;
  let w = origW, h = origH;
  if (w * h > MAX_INPUT_PIXELS) {
    const s = Math.sqrt(MAX_INPUT_PIXELS / (w * h));
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

function drawToCanvas(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext('2d').putImageData(
    new ImageData(imageData.data, imageData.width, imageData.height), 0, 0);
}

function hex(u8) {
  return Array.from(u8, b => b.toString(16).padStart(2, '0')).join('');
}

// Parse user-supplied bits. Accepts 896 chars of 0/1, or 224 hex chars.
function parseCustomBits(text) {
  const t = text.trim().replace(/\s+/g, '');
  if (/^[01]+$/.test(t)) {
    if (t.length !== N_BITS) {
      throw new Error(`expected ${N_BITS} 0/1 chars, got ${t.length}`);
    }
    const bits = new Uint8Array(N_BITS);
    for (let i = 0; i < N_BITS; i++) bits[i] = t.charCodeAt(i) - 48;
    return bits;
  }
  if (/^[0-9a-fA-F]+$/.test(t)) {
    const need = N_BITS / 4;
    if (t.length !== need) {
      throw new Error(`expected ${need} hex chars (${N_BITS} bits), got ${t.length}`);
    }
    const bits = new Uint8Array(N_BITS);
    for (let i = 0; i < t.length / 2; i++) {
      const b = parseInt(t.slice(i*2, i*2+2), 16);
      for (let j = 0; j < 8; j++) bits[i*8 + j] = (b >> (7 - j)) & 1;
    }
    return bits;
  }
  throw new Error(`bits must be ${N_BITS} 0/1 chars or ${N_BITS/4} hex chars`);
}

// ============================================================================
// ENCODE PANE
// ============================================================================
function initEncodePane() {
  els['enc-file'].addEventListener('change', e => loadEncFile(e.target.files?.[0]));
  els['enc-upload'].addEventListener('dragover', e => {
    e.preventDefault(); els['enc-upload'].classList.add('is-dragover');
  });
  els['enc-upload'].addEventListener('dragleave', () =>
    els['enc-upload'].classList.remove('is-dragover'));
  els['enc-upload'].addEventListener('drop', e => {
    e.preventDefault();
    els['enc-upload'].classList.remove('is-dragover');
    loadEncFile(e.dataTransfer.files?.[0]);
  });
  els['enc-sample'].addEventListener('click', e => { e.preventDefault(); loadEncSample(); });
  els['enc-runBtn'].addEventListener('click', runEncode);
  els['enc-downloadBtn'].addEventListener('click', downloadContainer);

  document.querySelectorAll('input[name="ih-bits-source"]').forEach(r => {
    r.addEventListener('change', () => {
      const checked = document.querySelector('input[name="ih-bits-source"]:checked').value;
      els['enc-customBits'].disabled = (checked !== 'custom');
    });
  });
}

async function loadEncFile(file) {
  if (!file) return;
  setEncStatus(`Loading ${file.name}…`);
  try {
    const bitmap = await createImageBitmap(file);
    onEncImage(bitmapToFitted(bitmap));
  } catch (e) {
    setEncStatus(`Failed to load: ${e.message}`, true);
  }
}

async function loadEncSample() {
  setEncStatus('Loading sample…');
  try {
    const resp = await fetch('/assets/imagehide/sample-cover.jpg');
    const blob = await resp.blob();
    const bitmap = await createImageBitmap(blob);
    onEncImage(bitmapToFitted(bitmap));
  } catch (e) {
    setEncStatus(`Failed to load sample: ${e.message}`, true);
  }
}

function onEncImage({ imageData, origW, origH }) {
  enc.coverImage = imageData;
  enc.origW = origW; enc.origH = origH;
  enc.crop = computeCrop(imageData.height, imageData.width);
  drawToCanvas(els.cover, imageData);
  const downNote = (origW !== imageData.width || origH !== imageData.height)
    ? ` (downsampled from ${origW}×${origH} — ${(origW * origH / 1e6).toFixed(1)} MP — to fit memory)`
    : '';
  setEncStatus(`Loaded ${imageData.width}×${imageData.height}${downNote}. Will encode ${enc.crop.cropW}×${enc.crop.cropH} (multiple of 64).`);
  refreshEncRunBtn();
}

function refreshEncRunBtn() {
  els['enc-runBtn'].disabled = !(modelStatus === 'ready' && enc.coverImage && !enc.busy);
}

async function runEncode() {
  if (enc.busy) return;
  enc.busy = true; refreshEncRunBtn();
  setEncStatus('Encoding…');

  try {
    const source = document.querySelector('input[name="ih-bits-source"]:checked').value;
    const { core, strips } = splitTrim(enc.coverImage, enc.crop);

    let H_bytes, sig, pk, bits, parts;
    if (source === 'auto') {
      H_bytes = phash128(core);
      sig = sodium.crypto_sign_detached(H_bytes, demoKeypair.privateKey);
      pk = demoKeypair.publicKey;
      bits = packPayload(H_bytes, sig, pk);
      parts = { H: H_bytes, sig, pk };
    } else {
      bits = parseCustomBits(els['enc-customBits'].value);
      const bytes = bitsToBytes(bits);
      parts = {
        H:   bytes.slice(0, N_H / 8),
        sig: bytes.slice(N_H / 8, (N_H + N_SIG) / 8),
        pk:  bytes.slice((N_H + N_SIG) / 8),
      };
    }

    const { container, ms } = await encode(core, bits);
    const psnrV = psnr(container, core);
    const ssimV = ssim(container, core);

    const fullContainer = pasteBack(container, strips, enc.crop,
                                    enc.coverImage.width, enc.coverImage.height);
    drawToCanvas(els.container, fullContainer);
    drawResidual(els.residual, core, container, 10);

    // Carry across to decode mode.
    lastContainer = fullContainer;
    lastPayloadBits = bits;
    refreshDecodeSource();

    // Release the encoder worker before decode mode might spawn the decoder.
    releaseSession();

    const psnrStr = isFinite(psnrV) ? `${psnrV.toFixed(1)} dB` : '∞ dB';
    const cropMsg = (enc.crop.trimmedTop || enc.crop.trimmedLeft)
      ? ` (${enc.crop.trimmedTop + enc.crop.trimmedBottom} px trimmed vertically, ${enc.crop.trimmedLeft + enc.crop.trimmedRight} px horizontally)`
      : '';
    els.oneshot.textContent =
      `Image: ${enc.origW} × ${enc.origH} → encoded region ${enc.crop.cropW} × ${enc.crop.cropH}${cropMsg}\n` +
      `Payload source: ${source}\n` +
      `Container vs cover: PSNR ${psnrStr} · SSIM ${ssimV.toFixed(4)}\n` +
      `Encode: ${ms.toFixed(1)} ms (decode ≈ same — same INN run in reverse)\n` +
      `\n` +
      `H   (128b): ${hex(parts.H)}\n` +
      `sig (512b): ${hex(parts.sig)}\n` +
      `pk  (256b): ${hex(parts.pk)}\n` +
      `\n` +
      `bits (MSB-first, ${bits.length}): ${Array.from(bits).join('')}`;

    els['enc-downloadBtn'].disabled = false;
    setEncStatus('Done. Switch to the Decode tab to recover the bits.');
  } catch (e) {
    setEncStatus(`Encode failed: ${e.message}`, true);
  } finally {
    enc.busy = false; refreshEncRunBtn();
  }
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

async function downloadContainer() {
  if (!lastContainer) return;
  const c = new OffscreenCanvas(lastContainer.width, lastContainer.height);
  c.getContext('2d').putImageData(
    new ImageData(lastContainer.data, lastContainer.width, lastContainer.height), 0, 0);
  const blob = await c.convertToBlob({ type: 'image/png' });
  c.width = 0; c.height = 0;
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'container.png';
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

// ============================================================================
// DECODE PANE
// ============================================================================
function initDecodePane() {
  // Attack dropdown: catalog + an explicit "none" entry.
  const opts = [
    { id: 'none', label: 'none (raw decode)' },
    ...ATTACKS.filter(a => a.id !== 'identity'),
  ];
  els['dec-attack'].innerHTML = opts.map(o =>
    `<option value="${o.id}">${o.label}</option>`).join('');

  els['dec-file'].addEventListener('change', e => loadDecFile(e.target.files?.[0]));
  els['dec-srcUpload'].addEventListener('change', refreshDecodeSource);
  els['dec-srcLast'].addEventListener('change', refreshDecodeSource);
  els['dec-runBtn'].addEventListener('click', runDecode);
  refreshDecodeSource();
}

function refreshDecodeSource() {
  // "Use last container" radio enabled iff we have one.
  els['dec-srcLast'].disabled = !lastContainer;
  els['dec-lastHint'].textContent = lastContainer
    ? ` — ${lastContainer.width}×${lastContainer.height}`
    : ' — none yet; encode one first or upload below';
  // Pick the right default if the user hasn't actively chosen.
  if (lastContainer && !els['dec-srcUpload'].checked) {
    els['dec-srcLast'].checked = true;
  }
  if (!lastContainer && !dec.uploadImage) {
    els['dec-srcUpload'].checked = true;
  }
  refreshDecRunBtn();
}

async function loadDecFile(file) {
  if (!file) return;
  setDecStatus(`Loading ${file.name}…`);
  try {
    const bitmap = await createImageBitmap(file);
    const { imageData, origW, origH } = bitmapToFitted(bitmap);
    dec.uploadImage = imageData;
    els['dec-srcUpload'].checked = true;
    const downNote = (origW !== imageData.width || origH !== imageData.height)
      ? ` (downsampled from ${origW}×${origH})` : '';
    setDecStatus(`Loaded ${imageData.width}×${imageData.height}${downNote}. Pick an attack and run.`);
    refreshDecRunBtn();
  } catch (e) {
    setDecStatus(`Failed to load: ${e.message}`, true);
  }
}

function refreshDecRunBtn() {
  if (modelStatus !== 'ready' || dec.busy) {
    els['dec-runBtn'].disabled = true; return;
  }
  const src = document.querySelector('input[name="ih-dec-source"]:checked').value;
  const have = (src === 'last' && lastContainer) || (src === 'upload' && dec.uploadImage);
  els['dec-runBtn'].disabled = !have;
}

async function runDecode() {
  if (dec.busy) return;
  dec.busy = true; refreshDecRunBtn();

  try {
    const src = document.querySelector('input[name="ih-dec-source"]:checked').value;
    const original = src === 'last' ? lastContainer : dec.uploadImage;
    if (!original) throw new Error('no image selected');

    // Decoder expects dimensions that are a multiple of 64. Crop centered.
    const crop = computeCrop(original.height, original.width);
    const { core } = splitTrim(original, crop);

    const attackId = els['dec-attack'].value;
    let attacked = core;
    let attackLabel = 'none';
    if (attackId !== 'none') {
      const a = ATTACKS.find(x => x.id === attackId);
      if (!a) throw new Error(`unknown attack: ${attackId}`);
      attackLabel = a.label;
      setDecStatus(`Applying ${attackLabel}…`);
      attacked = await a.fn(core);
    }

    drawToCanvas(els['dec-preview'], attacked);
    els['dec-previewLabel'].textContent = attackId === 'none'
      ? 'decoded image (no attack)'
      : `decoded image (after ${attackLabel})`;

    setDecStatus(`Decoding ${attacked.width}×${attacked.height}…`);
    const { bits: recBits, ms } = await decode(attacked);
    releaseSession();

    const { H: recH, sig: recSig, pk: recPk } = unpackPayload(recBits);
    let sigOk = false;
    try {
      sigOk = sodium.crypto_sign_verify_detached(recSig, recH, recPk);
    } catch (_) { sigOk = false; }

    const knownBits = (src === 'last') ? lastPayloadBits : null;
    const acc = knownBits ? bitAccuracy(recBits, knownBits) : null;

    els['dec-output'].textContent =
      `Image: ${original.width} × ${original.height} → decoded region ${attacked.width} × ${attacked.height}\n` +
      `Attack: ${attackLabel}\n` +
      `Decode: ${ms.toFixed(1)} ms\n` +
      `\n` +
      `Recovered:\n` +
      `  H   (128b): ${hex(recH)}\n` +
      `  sig (512b): ${hex(recSig)}\n` +
      `  pk  (256b): ${hex(recPk)}\n` +
      `\n` +
      `Bit accuracy vs known payload: ${acc != null ? acc.toFixed(4) : '— (no known reference)'}\n` +
      `Signature verifies under recovered pk: ${sigOk ? '✓ yes' : '✗ no'}\n` +
      `\n` +
      `bits (MSB-first, ${recBits.length}): ${Array.from(recBits).join('')}`;

    setDecStatus(`Done. Decoded in ${ms.toFixed(0)} ms.`);
  } catch (e) {
    setDecStatus(`Decode failed: ${e.message}`, true);
  } finally {
    dec.busy = false; refreshDecRunBtn();
  }
}
