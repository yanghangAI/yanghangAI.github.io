// Versioned dynamic imports so a redeploy busts every cached sibling.
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
const $ = (id) => document.getElementById(`ih-${id}`);
let sodium = null, demoKeypair = null;
let modelStatus = 'idle';

let lastContainer = null;
let lastPayloadBits = null;

const enc = { cover: null, origW: 0, origH: 0, crop: null, busy: false };
const dec = { upload: null, attackId: 'none', busy: false };

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else { init(); }

// ============================================================================
// INIT
// ============================================================================
function init() {
  initTabs();
  initEncodePane();
  initDecodePane();
  loadInfra();
}

function initTabs() {
  document.querySelectorAll('.ih-tab').forEach(btn => {
    btn.addEventListener('click', () => switchMode(btn.dataset.mode));
  });
}

function switchMode(mode) {
  document.querySelectorAll('.ih-tab').forEach(b => {
    const on = b.dataset.mode === mode;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-selected', on ? 'true' : 'false');
  });
  document.querySelectorAll('.ih-pane').forEach(p => {
    p.classList.toggle('is-hidden', p.dataset.pane !== mode);
  });
  releaseSession();
}

function setGlobalStatus(text, cls) {
  const el = $('globalStatus');
  el.textContent = text;
  el.className = 'ih-tabs__status' + (cls ? ' ' + cls : '');
}

function setRunStatus(which, html, isWarn = false) {
  const el = $(`${which}-status`);
  el.innerHTML = html;
  el.classList.toggle('is-warn', isWarn);
}

// ---------- model + sodium ----------
async function loadInfra() {
  modelStatus = 'loading';
  setGlobalStatus('loading model… 0%');
  try {
    await Promise.all([loadSodium(), loadModel()]);
    modelStatus = 'ready';
    setGlobalStatus(`ready · ${getBackend()}`, 'is-ready');
    setRunStatus('enc', 'Ready. Pick or drop a cover image.');
    setRunStatus('dec', 'Ready. Encode something first, or upload a watermarked image.');
    refreshEncRunBtn();
    refreshDecRunBtn();
    // Auto-prime the cover preview with the sample so a curious visitor can
    // click Encode immediately without uploading.
    if (!enc.cover) loadEncSample(true);
  } catch (e) {
    modelStatus = 'error';
    setGlobalStatus('model load failed', 'is-error');
    setRunStatus('enc', `Model load failed: ${e.message}`, true);
    setRunStatus('dec', `Model load failed: ${e.message}`, true);
  }
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
      setGlobalStatus(`loading model · ${pct}% (${(sum/1e6).toFixed(1)}/${(tot/1e6).toFixed(1)} MB)`);
    },
  );
}

// ---------- helpers ----------
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

function drawThumb(canvas, imageData, maxSize = 88) {
  const W = imageData.width, H = imageData.height;
  const s = Math.min(1, maxSize / Math.max(W, H));
  const tw = Math.max(1, Math.round(W * s));
  const th = Math.max(1, Math.round(H * s));
  canvas.width = tw; canvas.height = th;
  const off = new OffscreenCanvas(W, H);
  off.getContext('2d').putImageData(
    new ImageData(imageData.data, W, H), 0, 0);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'medium';
  ctx.drawImage(off, 0, 0, tw, th);
  off.width = 0; off.height = 0;
}

function hex(u8) {
  return Array.from(u8, b => b.toString(16).padStart(2, '0')).join('');
}

function parseCustomBits(text) {
  const t = text.trim().replace(/\s+/g, '');
  if (!t) throw new Error('paste some bits first');
  if (/^[01]+$/.test(t)) {
    if (t.length !== N_BITS) throw new Error(`expected ${N_BITS} 0/1 chars, got ${t.length}`);
    const bits = new Uint8Array(N_BITS);
    for (let i = 0; i < N_BITS; i++) bits[i] = t.charCodeAt(i) - 48;
    return bits;
  }
  if (/^[0-9a-fA-F]+$/.test(t)) {
    const need = N_BITS / 4;
    if (t.length !== need) throw new Error(`expected ${need} hex chars, got ${t.length}`);
    const bits = new Uint8Array(N_BITS);
    for (let i = 0; i < t.length / 2; i++) {
      const b = parseInt(t.slice(i*2, i*2+2), 16);
      for (let j = 0; j < 8; j++) bits[i*8 + j] = (b >> (7 - j)) & 1;
    }
    return bits;
  }
  throw new Error(`bits must be ${N_BITS} 0/1 chars or ${N_BITS/4} hex chars`);
}

// Generic radio "segmented" sync helper.
function bindSegmented(name, onChange) {
  document.querySelectorAll(`input[name="${name}"]`).forEach(r => {
    r.addEventListener('change', () => {
      document.querySelectorAll(`.ih-segmented__opt`).forEach(o => {
        const input = o.querySelector(`input[name="${name}"]`);
        if (input) o.classList.toggle('is-active', input.checked);
      });
      onChange && onChange(document.querySelector(`input[name="${name}"]:checked`).value);
    });
  });
}

// ============================================================================
// ENCODE PANE
// ============================================================================
function initEncodePane() {
  const drop = $('enc-drop');
  $('enc-file').addEventListener('change', e => loadEncFile(e.target.files?.[0]));
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('is-dragover'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('is-dragover'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('is-dragover');
    loadEncFile(e.dataTransfer.files?.[0]);
  });
  $('enc-sample').addEventListener('click', e => { e.preventDefault(); loadEncSample(false); });
  $('enc-thumb-clear').addEventListener('click', clearEncCover);

  $('enc-runBtn').addEventListener('click', runEncode);
  $('enc-downloadBtn').addEventListener('click', downloadContainer);
  $('enc-toDecodeBtn').addEventListener('click', () => {
    switchMode('decode');
    // Default to "last container" if available.
    $('dec-srcLast').checked = true;
    refreshDecodeSource();
  });

  bindSegmented('ih-bits-source', (val) => {
    $('enc-customBits').classList.toggle('is-hidden', val !== 'custom');
  });
}

async function loadEncFile(file) {
  if (!file) return;
  setRunStatus('enc', `Loading ${file.name}…`);
  try {
    const bitmap = await createImageBitmap(file);
    onEncImage(bitmapToFitted(bitmap), file.name);
  } catch (e) {
    setRunStatus('enc', `Failed to load: ${e.message}`, true);
  }
}

async function loadEncSample(silent) {
  if (!silent) setRunStatus('enc', 'Loading sample…');
  try {
    const resp = await fetch('/assets/imagehide/sample-cover.jpg');
    const blob = await resp.blob();
    const bitmap = await createImageBitmap(blob);
    onEncImage(bitmapToFitted(bitmap), 'sample-cover.jpg');
  } catch (e) {
    if (!silent) setRunStatus('enc', `Failed to load sample: ${e.message}`, true);
  }
}

function onEncImage({ imageData, origW, origH }, name) {
  enc.cover = imageData;
  enc.origW = origW; enc.origH = origH;
  enc.crop = computeCrop(imageData.height, imageData.width);

  drawThumb($('cover'), imageData);
  $('enc-thumb-name').textContent = `${name} · ${imageData.width}×${imageData.height}` +
    (origW !== imageData.width ? ` (from ${origW}×${origH})` : '');
  $('enc-thumb').classList.remove('is-hidden');
  $('enc-drop').querySelector('.ih-drop__cta').classList.add('is-hidden');

  setRunStatus('enc', `Will encode a ${enc.crop.cropW}×${enc.crop.cropH} region (multiple of 64).`);
  refreshEncRunBtn();
}

function clearEncCover(e) {
  e?.preventDefault?.();
  enc.cover = null; enc.crop = null;
  $('enc-thumb').classList.add('is-hidden');
  $('enc-drop').querySelector('.ih-drop__cta').classList.remove('is-hidden');
  $('enc-file').value = '';
  setRunStatus('enc', 'Pick or drop a cover image.');
  refreshEncRunBtn();
}

function refreshEncRunBtn() {
  $('enc-runBtn').disabled = !(modelStatus === 'ready' && enc.cover && !enc.busy);
}

async function runEncode() {
  if (enc.busy) return;
  enc.busy = true; refreshEncRunBtn();
  setRunStatus('enc', 'Encoding…');

  try {
    const source = document.querySelector('input[name="ih-bits-source"]:checked').value;
    const { core, strips } = splitTrim(enc.cover, enc.crop);

    let bits, parts;
    if (source === 'auto') {
      const H_bytes = phash128(core);
      const sig = sodium.crypto_sign_detached(H_bytes, demoKeypair.privateKey);
      const pk = demoKeypair.publicKey;
      bits = packPayload(H_bytes, sig, pk);
      parts = { H: H_bytes, sig, pk };
    } else {
      bits = parseCustomBits($('enc-customBits').value);
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
                                    enc.cover.width, enc.cover.height);

    lastContainer = fullContainer;
    lastPayloadBits = bits;

    drawToCanvas($('container'), fullContainer);
    drawResidual($('residual'), core, container, 10);

    $('m-psnr').textContent = isFinite(psnrV) ? `${psnrV.toFixed(1)}` : '∞';
    $('m-ssim').textContent = ssimV.toFixed(4);
    $('m-ms').textContent = `${ms.toFixed(0)} ms`;
    $('m-size').textContent = `${enc.crop.cropW}×${enc.crop.cropH}`;

    $('oneshot').textContent =
      `image: ${enc.origW}×${enc.origH} → encoded region ${enc.crop.cropW}×${enc.crop.cropH}\n` +
      `payload: ${source} (${N_BITS} bits = ${N_H} H | ${N_SIG} sig | 256 pk)\n` +
      `\n` +
      `H   : ${hex(parts.H)}\n` +
      `sig : ${hex(parts.sig)}\n` +
      `pk  : ${hex(parts.pk)}\n` +
      `\n` +
      `bits: ${Array.from(bits).join('')}`;

    $('enc-results').classList.remove('is-hidden');
    $('enc-results').scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    releaseSession();
    refreshDecodeSource();
    setRunStatus('enc', `Done · ${ms.toFixed(0)} ms.`);
  } catch (e) {
    setRunStatus('enc', `Encode failed: ${e.message}`, true);
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
  // Build attack chips: catalog + an explicit "none" entry up front.
  const opts = [
    { id: 'none', label: 'none' },
    ...ATTACKS.filter(a => a.id !== 'identity'),
  ];
  $('dec-attackChips').innerHTML = opts.map((o, i) =>
    `<button type="button" class="ih-chip ${i === 0 ? 'is-active' : ''}" data-attack="${o.id}">${o.label}</button>`
  ).join('');
  $('dec-attackChips').addEventListener('click', e => {
    const btn = e.target.closest('.ih-chip');
    if (!btn) return;
    $('dec-attackChips').querySelectorAll('.ih-chip').forEach(c => c.classList.remove('is-active'));
    btn.classList.add('is-active');
    dec.attackId = btn.dataset.attack;
  });

  $('dec-file').addEventListener('change', e => loadDecFile(e.target.files?.[0]));
  $('dec-runBtn').addEventListener('click', runDecode);

  bindSegmented('ih-dec-source', (val) => {
    $('dec-drop').classList.toggle('is-hidden', val !== 'upload');
    refreshDecRunBtn();
  });

  refreshDecodeSource();
}

function refreshDecodeSource() {
  const lastOk = !!lastContainer;
  $('dec-srcLast').disabled = !lastOk;
  $('dec-lastHint').textContent = lastOk
    ? `${lastContainer.width}×${lastContainer.height}, just encoded`
    : 'none yet — encode one first';
  // Sync segmented .is-active classes from input state.
  document.querySelectorAll('input[name="ih-dec-source"]').forEach(r => {
    const opt = r.closest('.ih-segmented__opt');
    if (opt) opt.classList.toggle('is-active', r.checked);
  });
  if (lastOk && !document.querySelector('input[name="ih-dec-source"]:checked')) {
    $('dec-srcLast').checked = true;
  }
  if (!lastOk && !dec.upload) {
    $('dec-srcUpload').checked = true;
  }
  // Refresh active highlighting one more time.
  document.querySelectorAll('input[name="ih-dec-source"]').forEach(r => {
    const opt = r.closest('.ih-segmented__opt');
    if (opt) opt.classList.toggle('is-active', r.checked);
  });
  $('dec-drop').classList.toggle('is-hidden',
    document.querySelector('input[name="ih-dec-source"]:checked')?.value !== 'upload');
  refreshDecRunBtn();
}

async function loadDecFile(file) {
  if (!file) return;
  setRunStatus('dec', `Loading ${file.name}…`);
  try {
    const bitmap = await createImageBitmap(file);
    const { imageData, origW, origH } = bitmapToFitted(bitmap);
    dec.upload = imageData;
    $('dec-srcUpload').checked = true;
    refreshDecodeSource();
    const downNote = (origW !== imageData.width || origH !== imageData.height)
      ? ` (downsampled from ${origW}×${origH})` : '';
    setRunStatus('dec', `Loaded ${imageData.width}×${imageData.height}${downNote}.`);
  } catch (e) {
    setRunStatus('dec', `Failed to load: ${e.message}`, true);
  }
}

function refreshDecRunBtn() {
  if (modelStatus !== 'ready' || dec.busy) {
    $('dec-runBtn').disabled = true; return;
  }
  const src = document.querySelector('input[name="ih-dec-source"]:checked')?.value;
  const have = (src === 'last' && lastContainer) || (src === 'upload' && dec.upload);
  $('dec-runBtn').disabled = !have;
}

async function runDecode() {
  if (dec.busy) return;
  dec.busy = true; refreshDecRunBtn();

  try {
    const src = document.querySelector('input[name="ih-dec-source"]:checked').value;
    const original = src === 'last' ? lastContainer : dec.upload;
    if (!original) throw new Error('no image selected');

    const crop = computeCrop(original.height, original.width);
    const { core } = splitTrim(original, crop);

    let attacked = core;
    let attackLabel = 'none';
    if (dec.attackId !== 'none') {
      const a = ATTACKS.find(x => x.id === dec.attackId);
      if (!a) throw new Error(`unknown attack: ${dec.attackId}`);
      attackLabel = a.label;
      setRunStatus('dec', `Applying ${attackLabel}…`);
      attacked = await a.fn(core);
    }

    drawToCanvas($('dec-preview'), attacked);
    $('dec-previewLabel').textContent = dec.attackId === 'none'
      ? 'decoded image (no attack)'
      : `decoded image (after ${attackLabel})`;

    setRunStatus('dec', `Decoding ${attacked.width}×${attacked.height}…`);
    const { bits: recBits, ms } = await decode(attacked);
    releaseSession();

    const { H: recH, sig: recSig, pk: recPk } = unpackPayload(recBits);
    let sigOk = false;
    try { sigOk = sodium.crypto_sign_verify_detached(recSig, recH, recPk); }
    catch (_) { sigOk = false; }

    const knownBits = (src === 'last') ? lastPayloadBits : null;
    const acc = knownBits ? bitAccuracy(recBits, knownBits) : null;

    const accEl = $('d-acc');
    if (acc != null) {
      accEl.textContent = acc.toFixed(4);
      accEl.classList.toggle('is-ok', acc >= 0.95);
      accEl.classList.toggle('is-bad', acc < 0.80);
    } else {
      accEl.textContent = '—';
      accEl.classList.remove('is-ok', 'is-bad');
    }
    const sigEl = $('d-sig');
    sigEl.textContent = sigOk ? '✓' : '✗';
    sigEl.classList.toggle('is-ok', sigOk);
    sigEl.classList.toggle('is-bad', !sigOk);
    $('d-ms').textContent = `${ms.toFixed(0)} ms`;
    $('d-atk').textContent = attackLabel;

    $('dec-output').textContent =
      `image: ${original.width}×${original.height} → decoded region ${attacked.width}×${attacked.height}\n` +
      `attack: ${attackLabel}\n` +
      `\n` +
      `recovered:\n` +
      `  H   : ${hex(recH)}\n` +
      `  sig : ${hex(recSig)}\n` +
      `  pk  : ${hex(recPk)}\n` +
      `\n` +
      `bit accuracy vs known payload: ${acc != null ? acc.toFixed(4) : '— (no reference)'}\n` +
      `signature verifies under recovered pk: ${sigOk ? 'yes' : 'no'}\n` +
      `\n` +
      `bits: ${Array.from(recBits).join('')}`;

    $('dec-results').classList.remove('is-hidden');
    $('dec-results').scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    setRunStatus('dec', `Done · ${ms.toFixed(0)} ms.`);
  } catch (e) {
    setRunStatus('dec', `Decode failed: ${e.message}`, true);
  } finally {
    dec.busy = false; refreshDecRunBtn();
  }
}
