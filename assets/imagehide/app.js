// Versioned dynamic imports so a redeploy busts cached siblings.
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

const $ = (id) => document.getElementById(`ih-${id}`);

let sodium = null, demoKeypair = null;
let modelStatus = 'idle';

let lastContainer = null;
let lastPayloadBits = null;

const enc = { cover: null, origW: 0, origH: 0, crop: null, busy: false };
const dec = { upload: null, busy: false };

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else { init(); }

function init() {
  initEncode();
  initDecode();
  loadInfra();
}

// ---------- model + sodium ----------
async function loadInfra() {
  modelStatus = 'loading';
  setGlobalBar('loading model…');
  try {
    await Promise.all([loadSodium(), loadModel()]);
    modelStatus = 'ready';
    setGlobalBar(`ready · running on ${getBackend()}${IS_MOBILE ? ' (mobile cap 0.5 MP)' : ' (desktop cap 1 MP)'}`, 'is-ready');
    setStatus('enc', 'Ready. Drop a cover image, or use the sample.');
    setStatus('dec', 'Ready. Encode something first, or upload a watermarked image.');
    refreshEncRun(); refreshDecRun();
    if (!enc.cover) loadEncSample(true);
  } catch (e) {
    modelStatus = 'error';
    setGlobalBar(`model load failed: ${e.message}`, 'is-error');
    setStatus('enc', `Model load failed: ${e.message}`, true);
    setStatus('dec', `Model load failed: ${e.message}`, true);
  }
}

async function loadSodium() {
  if (sodium) return;
  const mod = await import(/* @vite-ignore */ LIBSODIUM_CDN);
  sodium = mod.default || mod;
  await sodium.ready;
  demoKeypair = sodium.crypto_sign_keypair();
}

const mp = { encoder: 0, decoder: 0, total: { encoder: 0, decoder: 0 } };
async function loadModel() {
  await loadModels(
    '/assets/imagehide/encoder.onnx',
    '/assets/imagehide/decoder.onnx',
    ({ tag, loaded, total }) => {
      mp[tag] = loaded;
      if (total) mp.total[tag] = total;
      const sum = mp.encoder + mp.decoder;
      const tot = mp.total.encoder + mp.total.decoder;
      const pct = tot ? Math.round(100 * sum / tot) : 0;
      setGlobalBar(`loading model · ${pct}% (${(sum/1e6).toFixed(1)}/${(tot/1e6).toFixed(1)} MB)`);
    },
  );
}

// ---------- helpers ----------
let _baseStatus = '';
let _baseStatusCls = '';
function setGlobalBar(text, cls = '') {
  _baseStatus = text;
  _baseStatusCls = cls;
  renderGlobalBar();
}

// Live JS heap readout. Chrome exposes performance.memory; Safari and Firefox
// do not (returns undefined), so we show "heap n/a" on those. WASM heap (worker
// side) is NOT visible from main thread — but main-thread JS heap movement
// still tracks ImageData and Canvas memory pressure which is the bulk of the
// non-WASM cost.
function fmtMB(bytes) { return `${(bytes / 1048576).toFixed(0)} MB`; }
function heapSuffix() {
  const m = performance.memory;
  if (!m) return ' · heap n/a (Safari)';
  return ` · heap ${fmtMB(m.usedJSHeapSize)} / ${fmtMB(m.jsHeapSizeLimit)}`;
}
function renderGlobalBar() {
  const el = document.getElementById('ih-globalbar');
  const txt = document.getElementById('ih-globalbar-text');
  if (!el || !txt) return;
  txt.textContent = _baseStatus + heapSuffix();
  el.className = 'ih-globalbar' + (_baseStatusCls ? ' ' + _baseStatusCls : '');
}
// Refresh the heap reading periodically. The interval is cheap.
setInterval(renderGlobalBar, 1500);

function setStatus(which, html, isWarn = false) {
  const el = $(`${which}-status`);
  el.innerHTML = html;
  el.classList.toggle('is-warn', isWarn);
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

function drawThumb(canvas, imageData, sz = 88) {
  const W = imageData.width, H = imageData.height;
  const off = new OffscreenCanvas(W, H);
  off.getContext('2d').putImageData(
    new ImageData(imageData.data, W, H), 0, 0);
  canvas.width = sz; canvas.height = sz;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'medium';
  // Cover-fit (crop center).
  const scale = Math.max(sz / W, sz / H);
  const dw = W * scale, dh = H * scale;
  ctx.drawImage(off, (sz - dw)/2, (sz - dh)/2, dw, dh);
  off.width = 0; off.height = 0;
}

function hex(u8) {
  return Array.from(u8, b => b.toString(16).padStart(2, '0')).join('');
}

// Format an 896-bit payload as a fixed grid: 14 lines of 64 bits, grouped in
// octets separated by single spaces. Identical structure in both cards makes
// the bit positions land at the same column / line on both sides.
function chunkBits(bits) {
  const arr = Array.from(bits);
  const lines = [];
  for (let i = 0; i < arr.length; i += 64) {
    const groups = [];
    for (let j = i; j < Math.min(i + 64, arr.length); j += 8) {
      groups.push(arr.slice(j, j + 8).join(''));
    }
    lines.push(groups.join(' '));
  }
  return lines.join('\n');
}

// Same layout as chunkBits, but each bit becomes a <span> when it disagrees
// with the known reference. Used in the decode card when we know the original
// payload (i.e. when src='last container').
function chunkBitsHTML(bits, known) {
  const arr = Array.from(bits);
  const ref = known ? Array.from(known) : null;
  const lines = [];
  for (let i = 0; i < arr.length; i += 64) {
    const groups = [];
    for (let j = i; j < Math.min(i + 64, arr.length); j += 8) {
      const grp = [];
      for (let k = j; k < Math.min(j + 8, arr.length); k++) {
        if (ref && arr[k] !== ref[k]) {
          grp.push(`<span class="b-bad">${arr[k]}</span>`);
        } else {
          grp.push(String(arr[k]));
        }
      }
      groups.push(grp.join(''));
    }
    lines.push(groups.join(' '));
  }
  return lines.join('\n');
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
  throw new Error(`paste ${N_BITS} 0/1 chars or ${N_BITS/4} hex chars`);
}

// ===========================================================================
// ENCODE
// ===========================================================================
function initEncode() {
  const drop = $('enc-drop');
  $('enc-file').addEventListener('change', e => loadEncFile(e.target.files?.[0]));
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('is-dragover'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('is-dragover'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('is-dragover');
    loadEncFile(e.dataTransfer.files?.[0]);
  });
  $('enc-sample').addEventListener('click', e => { e.preventDefault(); loadEncSample(false); });
  $('enc-clear').addEventListener('click', clearEncCover);
  $('enc-runBtn').addEventListener('click', runEncode);
  $('enc-downloadBtn').addEventListener('click', downloadContainer);
  $('enc-toDecodeBtn').addEventListener('click', sendToDecode);

  document.querySelectorAll('input[name="ih-bits-source"]').forEach(r => {
    r.addEventListener('change', () => {
      const val = document.querySelector('input[name="ih-bits-source"]:checked').value;
      $('enc-customBits').classList.toggle('is-hidden', val !== 'custom');
    });
  });
}

async function loadEncFile(file) {
  if (!file) return;
  setStatus('enc', `Loading ${file.name}…`);
  try {
    const bitmap = await createImageBitmap(file);
    onEncImage(bitmapToFitted(bitmap), file.name);
  } catch (e) {
    setStatus('enc', `Failed to load: ${e.message}`, true);
  }
}

async function loadEncSample(silent) {
  if (!silent) setStatus('enc', 'Loading sample…');
  try {
    const resp = await fetch('/assets/imagehide/sample-cover.jpg');
    const blob = await resp.blob();
    const bitmap = await createImageBitmap(blob);
    onEncImage(bitmapToFitted(bitmap), 'sample.jpg');
  } catch (e) {
    if (!silent) setStatus('enc', `Failed to load sample: ${e.message}`, true);
  }
}

function onEncImage({ imageData, origW, origH }, name) {
  enc.cover = imageData;
  enc.origW = origW; enc.origH = origH;
  enc.crop = computeCrop(imageData.height, imageData.width);

  drawToCanvas($('cover'), imageData);
  const dim = `${imageData.width}×${imageData.height}` +
    (origW !== imageData.width ? ` · from ${origW}×${origH}` : '');
  $('enc-name').textContent = `${name} · ${dim}`;
  $('enc-preview').classList.remove('is-hidden');
  $('enc-drop').classList.add('is-hidden');

  setStatus('enc', `Will encode ${enc.crop.cropW}×${enc.crop.cropH} region. Click Encode.`);
  refreshEncRun();
}

function clearEncCover(e) {
  e?.preventDefault?.();
  enc.cover = null; enc.crop = null;
  $('enc-preview').classList.add('is-hidden');
  $('enc-drop').classList.remove('is-hidden');
  $('enc-file').value = '';
  setStatus('enc', 'Pick or drop a cover image.');
  refreshEncRun();
}

function refreshEncRun() {
  $('enc-runBtn').disabled = !(modelStatus === 'ready' && enc.cover && !enc.busy);
}

async function runEncode() {
  if (enc.busy) return;
  enc.busy = true; refreshEncRun();
  setStatus('enc', 'Encoding…');

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

    $('m-psnr').textContent = isFinite(psnrV) ? psnrV.toFixed(1) : '∞';
    $('m-ssim').textContent = ssimV.toFixed(4);
    $('m-ms').textContent = `${ms.toFixed(0)} ms`;

    // Format matches the decode card line-for-line so the two codeblocks line up
    // when the user opens both <details>.
    $('oneshot').textContent =
      `image:   ${enc.origW}×${enc.origH} → encoded region ${enc.crop.cropW}×${enc.crop.cropH}\n` +
      `payload: ${source} (${N_BITS} bits)\n` +
      `\n` +
      `H   : ${hex(parts.H)}\n` +
      `sig : ${hex(parts.sig)}\n` +
      `pk  : ${hex(parts.pk)}\n` +
      `\n` +
      `bits:\n${chunkBits(bits)}`;

    $('enc-results').classList.remove('is-hidden');

    releaseSession();
    refreshDecSource();
    setStatus('enc', `Done · ${ms.toFixed(0)} ms.`);
  } catch (e) {
    setStatus('enc', `Encode failed: ${e.message}`, true);
  } finally {
    enc.busy = false; refreshEncRun();
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

function sendToDecode() {
  if (!lastContainer) return;
  $('dec-srcLast').checked = true;
  refreshDecSource();
  const col = document.querySelector('.ih-col[data-col="decode"]');
  col.classList.remove('is-flash');
  void col.offsetWidth;
  col.classList.add('is-flash');
  col.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ===========================================================================
// DECODE
// ===========================================================================
function initDecode() {
  const opts = [
    { id: 'none', label: 'none (raw decode)' },
    ...ATTACKS.filter(a => a.id !== 'identity'),
  ];
  $('dec-attack').innerHTML = opts.map(o =>
    `<option value="${o.id}">${o.label}</option>`).join('');

  $('dec-file').addEventListener('change', e => loadDecFile(e.target.files?.[0]));
  const drop = $('dec-drop');
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('is-dragover'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('is-dragover'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('is-dragover');
    loadDecFile(e.dataTransfer.files?.[0]);
  });

  document.querySelectorAll('input[name="ih-dec-source"]').forEach(r => {
    r.addEventListener('change', refreshDecSource);
  });
  $('dec-runBtn').addEventListener('click', runDecode);
  refreshDecSource();
}

function refreshDecSource() {
  const lastOk = !!lastContainer;
  $('dec-srcLast').disabled = !lastOk;
  const hint = $('dec-lastHint');
  if (hint) {
    hint.textContent = lastOk
      ? `— ${lastContainer.width}×${lastContainer.height}`
      : '— none yet, encode one first';
  }
  if (lastOk && !document.querySelector('input[name="ih-dec-source"]:checked')) {
    $('dec-srcLast').checked = true;
  }
  if (!lastOk && !document.querySelector('input[name="ih-dec-source"]:checked')) {
    $('dec-srcUpload').checked = true;
  }
  if (!lastOk && $('dec-srcLast').checked) $('dec-srcUpload').checked = true;
  const src = document.querySelector('input[name="ih-dec-source"]:checked')?.value;
  $('dec-drop').classList.toggle('is-hidden', src !== 'upload');
  updateDecPreview();
  refreshDecRun();
}

// Show the currently-chosen decode input as a 14rem preview, matching the
// cover preview on the encode side. This is the user's INPUT image — not the
// post-attack one. The attack effect is implicit in the dropdown selection.
function updateDecPreview() {
  const src = document.querySelector('input[name="ih-dec-source"]:checked')?.value;
  const wrap = $('dec-srcPreview');
  const canvas = $('dec-srcCanvas');
  const name = $('dec-srcName');
  let img = null, label = '';
  if (src === 'last' && lastContainer) {
    img = lastContainer;
    label = `last container · ${img.width}×${img.height}`;
  } else if (src === 'upload' && dec.upload) {
    img = dec.upload;
    label = `uploaded · ${img.width}×${img.height}`;
  }
  if (!img) { wrap.classList.add('is-hidden'); return; }
  drawToCanvas(canvas, img);
  name.textContent = label;
  wrap.classList.remove('is-hidden');
}

async function loadDecFile(file) {
  if (!file) return;
  setStatus('dec', `Loading ${file.name}…`);
  try {
    const bitmap = await createImageBitmap(file);
    const { imageData, origW, origH } = bitmapToFitted(bitmap);
    dec.upload = imageData;
    $('dec-srcUpload').checked = true;
    refreshDecSource();
    const note = (origW !== imageData.width)
      ? ` (from ${origW}×${origH})` : '';
    setStatus('dec', `Loaded ${imageData.width}×${imageData.height}${note}.`);
  } catch (e) {
    setStatus('dec', `Failed to load: ${e.message}`, true);
  }
}

function refreshDecRun() {
  if (modelStatus !== 'ready' || dec.busy) {
    $('dec-runBtn').disabled = true; return;
  }
  const src = document.querySelector('input[name="ih-dec-source"]:checked')?.value;
  const have = (src === 'last' && lastContainer) || (src === 'upload' && dec.upload);
  $('dec-runBtn').disabled = !have;
}

async function runDecode() {
  if (dec.busy) return;
  dec.busy = true; refreshDecRun();

  try {
    const src = document.querySelector('input[name="ih-dec-source"]:checked').value;
    const original = src === 'last' ? lastContainer : dec.upload;
    if (!original) throw new Error('no image selected');

    const crop = computeCrop(original.height, original.width);
    const { core } = splitTrim(original, crop);

    const attackId = $('dec-attack').value;
    let attacked = core;
    let attackLabel = 'none';
    if (attackId !== 'none') {
      const a = ATTACKS.find(x => x.id === attackId);
      if (!a) throw new Error(`unknown attack: ${attackId}`);
      attackLabel = a.label;
      setStatus('dec', `Applying ${attackLabel}…`);
      attacked = await a.fn(core);
    }

    setStatus('dec', `Decoding ${attacked.width}×${attacked.height}…`);
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
    $('d-ms').textContent = `${ms.toFixed(0)} ms`;

    // Same line structure as the encode card so the two codeblocks align;
    // the bits use innerHTML to wrap mismatches in <span class="b-bad"> when
    // a reference payload is known.
    const accLine = acc != null
      ? `bit acc: ${acc.toFixed(4)}  ·  sig: ${sigOk ? 'yes' : 'no'}`
      : `bit acc: — (no reference)  ·  sig: ${sigOk ? 'yes' : 'no'}`;
    const bitsHtml = chunkBitsHTML(recBits, knownBits);
    $('dec-output').innerHTML =
      `image:   ${original.width}×${original.height} → decoded region ${attacked.width}×${attacked.height}\n` +
      `attack:  ${attackLabel}\n` +
      `${accLine}\n` +
      `\n` +
      `H   : ${hex(recH)}\n` +
      `sig : ${hex(recSig)}\n` +
      `pk  : ${hex(recPk)}\n` +
      `\n` +
      `bits:\n${bitsHtml}`;

    $('dec-results').classList.remove('is-hidden');
    setStatus('dec', `Done · ${ms.toFixed(0)} ms.`);
  } catch (e) {
    setStatus('dec', `Decode failed: ${e.message}`, true);
  } finally {
    dec.busy = false; refreshDecRun();
  }
}
