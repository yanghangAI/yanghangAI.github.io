// Versioned dynamic imports so a redeploy busts cached siblings.
const V = (typeof window !== 'undefined' && window.__imagehideVersion) || 'dev';
const { computeCrop, splitTrim, pasteBack } = await import(`./trim.js?v=${V}`);
const { phash128, packPayload, unpackPayload, bitAccuracy, bitsToBytes, bytesToBits,
        N_H, N_SIG, N_PK, N_BITS } = await import(`./payload.js?v=${V}`);
const { psnr, ssim } = await import(`./metrics.js?v=${V}`);
const { ATTACKS, imageDataToFrame, frameToImageData } =
  await import(`./attacks.js?v=${V}`);
const { buildPermStack, dwtDims } = await import(`./perm.js?v=${V}`);
const { loadModels, encode, decode, getBackend, releaseSession } =
  await import(`./pipeline.js?v=${V}`);
// Outer ECC: RS(128, 100) over GF(2^8). 796 wire bits -> 1024-bit codeword
// (what the model sees), corrects up to 14 byte errors of channel damage.
const { eccEncode, eccDecode, T_BYTES, PAYLOAD_BITS, CODEWORD_BITS } =
  await import(`./ecc.js?v=${V}`);
// Slepian-Wolf pHash compression: BCH(127, t=4). We transmit a 28-bit
// syndrome of H instead of the full 128 bits; the receiver recomputes pHash
// on the attacked image and uses the syndrome to recover H exactly, fixing
// up to 4 bits of pHash drift (PDQ's observed max across 30 COCO is 3).
const { bchEncodeSyndrome, bchDecode, BCH_SYNDROME_BITS, BCH_T } =
  await import(`./bch.js?v=${V}`);

const MODEL_BITS = CODEWORD_BITS;     // 1024

// Wire payload layout (796 bits, exactly PAYLOAD_BITS):
//   [0   .. 28 )  BCH syndrome of pHash
//   [28  .. 540)  Ed25519 signature (512 bits)
//   [540 ..796 )  Ed25519 public key (256 bits)
const OFF_SYN = 0;
const OFF_SIG = OFF_SYN + BCH_SYNDROME_BITS;             // 49
const OFF_PK  = OFF_SIG + N_SIG;                         // 561

function packWirePayload(syndromeBits, sigBytes, pkBytes) {
  const out = new Uint8Array(PAYLOAD_BITS);
  out.set(syndromeBits, OFF_SYN);
  out.set(bytesToBits(sigBytes), OFF_SIG);
  out.set(bytesToBits(pkBytes),  OFF_PK);
  return out;
}
function unpackWirePayload(bits817) {
  return {
    syndrome: bits817.slice(OFF_SYN, OFF_SIG),
    sig:      bitsToBytes(bits817.slice(OFF_SIG, OFF_PK)),
    pk:       bitsToBytes(bits817.slice(OFF_PK,  OFF_PK + N_PK)),
  };
}
// pHash is 128 bits but BCH(127) carries 127. Drop the LSB of the last byte
// (force bit 127 = 0) so encoder and decoder agree on a canonical 16-byte H.
function canonicalizeH(H16) {
  const out = Uint8Array.from(H16);
  out[15] &= 0xFE;
  return out;
}

const LIBSODIUM_CDN = 'https://cdn.jsdelivr.net/npm/libsodium-wrappers@0.7.13/+esm';
const IS_MOBILE = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
// 1 MP cap on desktop and mobile — the decoder's forward-pass intermediates
// peak at ~60-120 MB per megapixel; 1 MP fits Safari/iOS's ~300 MB tab budget
// with the two-process encoder→decoder pipeline that releases the encoder
// worker before decoding starts.
const MAX_INPUT_PIXELS = 1 * 1024 * 1024;
// Minimum shorter-dimension after fit. The ONNX models were traced at H=W=256
// to bake the canonical pHash-adapter permutation; smaller inputs throw a
// ScatterElements out-of-range error at inference. We upscale anything below
// this threshold to keep the demo working on small thumbnails.
const MIN_SHORT_DIM = 256;

const $ = (id) => document.getElementById(`ih-${id}`);

let sodium = null, demoKeypair = null;
let modelStatus = 'idle';

let lastContainer = null;
let lastPayloadBits = null;     // 796-bit wire payload (syndrome|sig|pk)
let lastCodewordBits = null;    // 1024-bit RS codeword
let lastH = null;                // canonical 16-byte H (LSB of last byte cleared)
let lastSyndrome = null;         // 28-bit BCH syndrome
let lastLogicalParts = null;     // { H, sig, pk } for display + bit-acc
let lastSource = null;           // 'auto' | 'custom' — what payload mode encoded lastContainer
let lastContainerCoreF32 = null; // Float32Array CHW [-1,1] of the encoded core region
let lastContainerCoreW = 0;
let lastContainerCoreH = 0;

// Cache permutations by DWT size so multiple decodes on the same image don't
// recompute the (slow) MT19937 + argsort over hundreds of thousands of
// positions. Keyed by `${hDwt}x${wDwt}`.
const _permCache = new Map();
function permFor(inputH, inputW) {
  const { hDwt, wDwt } = dwtDims(inputH, inputW);
  const key = `${hDwt}x${wDwt}`;
  let perm = _permCache.get(key);
  if (!perm) {
    perm = buildPermStack(hDwt, wDwt);
    _permCache.set(key, perm);
  }
  return perm;
}

const enc = { cover: null, origW: 0, origH: 0, crop: null, busy: false };
const dec = { upload: null, busy: false };

// init() is invoked at the END of this file — see the bottom for the
// DOMContentLoaded handler. Moved there because invoking init() here triggers
// loadInfra() -> setGlobalBar() -> assignment to `_baseStatus` (a `let` declared
// further down), which throws a TDZ ReferenceError silently inside the async
// loadInfra body, leaving the page stuck on the bootstrap banner.

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
    setGlobalBar(`ready · running on ${getBackend()} · 1 MP cap`, 'is-ready');
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
  // Downscale if over the megapixel budget.
  if (w * h > MAX_INPUT_PIXELS) {
    const s = Math.sqrt(MAX_INPUT_PIXELS / (w * h));
    w = Math.max(64, Math.round(w * s));
    h = Math.max(64, Math.round(h * s));
  }
  // Upscale if the shorter side is below the ONNX 256-trace minimum. After
  // splitTrim rounds down to a multiple of 64 we need >=256 on each side, so
  // make the shorter dimension at least MIN_SHORT_DIM while preserving aspect.
  // We deliberately allow the resulting pixel count to exceed MAX_INPUT_PIXELS
  // here — for extreme aspect ratios (e.g. 100x2000) clamping again would
  // shrink the shorter dim back below 256 and fail at inference.
  const short = Math.min(w, h);
  if (short < MIN_SHORT_DIM) {
    const s = MIN_SHORT_DIM / short;
    w = Math.round(w * s);
    h = Math.round(h * s);
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

// 96 bytes = (sig + pk) slots — the H slot still holds the real pHash so the
// Slepian-Wolf BCH/RS pipeline keeps working unchanged. Text is UTF-8; the
// remaining bytes are zero-padded.
const CUSTOM_TEXT_BYTES = (N_SIG + N_PK) / 8;  // 96

function parseCustomText(text) {
  const buf = new TextEncoder().encode(text);
  if (buf.length > CUSTOM_TEXT_BYTES) {
    throw new Error(`message is ${buf.length} bytes, max is ${CUSTOM_TEXT_BYTES} (UTF-8)`);
  }
  const out = new Uint8Array(CUSTOM_TEXT_BYTES);
  out.set(buf, 0);
  return out;
}

// Decode a 96-byte block as UTF-8, stripping trailing NUL padding. Invalid
// UTF-8 sequences (e.g. random sig bytes in auto mode) are replaced with
// U+FFFD so the line is always renderable.
function bytesToCustomText(bytes96) {
  let end = bytes96.length;
  while (end > 0 && bytes96[end - 1] === 0) end--;
  return new TextDecoder('utf-8', { fatal: false }).decode(bytes96.subarray(0, end));
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
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
      $('enc-customWrap').classList.toggle('is-hidden', val !== 'custom');
    });
  });

  const ta = $('enc-customBits');
  const cc = $('enc-charcount');
  const updateCharCount = () => {
    const bytes = new TextEncoder().encode(ta.value).length;
    cc.textContent = `${bytes} / ${CUSTOM_TEXT_BYTES} bytes`;
    cc.classList.toggle('is-over', bytes > CUSTOM_TEXT_BYTES);
  };
  ta.addEventListener('input', updateCharCount);
  updateCharCount();
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
    if (enc.crop.cropW < 256 || enc.crop.cropH < 256) {
      throw new Error(
        `encoded region ${enc.crop.cropW}x${enc.crop.cropH} is below the ` +
        `256x256 minimum baked into the ONNX trace`);
    }
    const { core, strips } = splitTrim(enc.cover, enc.crop);

    let H, sig, pk;
    if (source === 'auto') {
      H = canonicalizeH(phash128(core));
      sig = sodium.crypto_sign_detached(H, demoKeypair.privateKey);
      pk = demoKeypair.publicKey;
    } else {
      // Custom-text mode: H stays as the real pHash (so BCH/Slepian-Wolf still
      // works), the user's UTF-8 message is packed into the sig+pk slots.
      H = canonicalizeH(phash128(core));
      const textBytes = parseCustomText($('enc-customBits').value);
      sig = textBytes.slice(0, N_SIG / 8);
      pk  = textBytes.slice(N_SIG / 8);
    }
    const parts = { H, sig, pk };

    // Slepian-Wolf: transmit only the BCH syndrome of H (49 bits) instead of
    // H itself (128 bits). Then wrap (syndrome | sig | pk) = 796 wire bits
    // in the RS(128, 100) codeword the model sees.
    const H_bits128 = bytesToBits(H);
    const syndrome = bchEncodeSyndrome(H_bits128.slice(0, 127));
    const wireBits = packWirePayload(syndrome, sig, pk);
    const codeword = eccEncode(wireBits);

    const encPerm = permFor(core.height, core.width);
    const { container, containerF32, ms } = await encode(core, codeword, encPerm);
    const psnrV = psnr(container, core);
    const ssimV = ssim(container, core);

    const fullContainer = pasteBack(container, strips, enc.crop,
                                    enc.cover.width, enc.cover.height);

    lastContainer = fullContainer;
    lastContainerCoreF32 = containerF32;
    lastContainerCoreW = container.width;
    lastContainerCoreH = container.height;
    lastPayloadBits = wireBits;
    lastCodewordBits = codeword;
    lastH = H;
    lastSyndrome = syndrome;
    lastLogicalParts = parts;
    lastSource = source;

    drawToCanvas($('container'), fullContainer);
    drawResidual($('residual'), core, container, 10);

    $('m-psnr').textContent = isFinite(psnrV) ? psnrV.toFixed(1) : '∞';
    $('m-ssim').textContent = ssimV.toFixed(4);
    $('m-ms').textContent = `${ms.toFixed(0)} ms`;

    const synStr = (() => {
      const g = [];
      for (let i = 0; i < BCH_T; i++) g.push(Array.from(syndrome.slice(i * 7, (i + 1) * 7)).join(''));
      return g.join(' ');
    })();
    const msgLine = source === 'custom'
      ? `message  : "${bytesToCustomText(new Uint8Array([...parts.sig, ...parts.pk]))}"  (${CUSTOM_TEXT_BYTES} bytes, packed into sig+pk slots)\n`
      : '';
    $('oneshot').textContent =
      `image:    ${enc.origW}×${enc.origH} → encoded region ${enc.crop.cropW}×${enc.crop.cropH}\n` +
      `payload:  ${source}  (Slepian-Wolf: ${BCH_SYNDROME_BITS}b BCH(t=${BCH_T}) syndrome + ${N_SIG}b sig + ${N_PK}b pk = ${PAYLOAD_BITS} wire → RS corrects ${T_BYTES} byte errors → ${MODEL_BITS}b codeword)\n` +
      `\n` +
      msgLine +
      `H        : ${hex(parts.H)}  (128 bits, recomputable from image)\n` +
      `BCH syn  : ${synStr}  (49 bits, embedded in place of H)\n` +
      `sig      : ${hex(parts.sig)}\n` +
      `pk       : ${hex(parts.pk)}\n` +
      `\n` +
      `codeword (${MODEL_BITS} bits embedded):\n${chunkBits(codeword)}`;

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
    if (crop.cropW < 256 || crop.cropH < 256) {
      throw new Error(
        `decode region ${crop.cropW}x${crop.cropH} is below the 256x256 ` +
        `minimum baked into the ONNX trace`);
    }

    // Build the Frame fed to the attack. For "last" we use the float32
    // container straight from the encoder — sub-uint8 residual survives the
    // resize step, matching the eval pipeline that keeps the container in
    // float32 between encoder and attack. For uploaded images we have no
    // float32 reference; convert from uint8 ImageData (no precision gain
    // there, but the resize algorithm at least matches PyTorch).
    let coreFrame, coreImageData;
    if (src === 'last' && lastContainerCoreF32 &&
        lastContainerCoreW === crop.cropW && lastContainerCoreH === crop.cropH) {
      coreFrame = {
        dataF32: lastContainerCoreF32,
        width:   lastContainerCoreW,
        height:  lastContainerCoreH,
      };
      coreImageData = frameToImageData(coreFrame);
    } else {
      coreImageData = splitTrim(original, crop).core;
      coreFrame = imageDataToFrame(coreImageData);
    }

    const attackId = $('dec-attack').value;
    let attacked = coreImageData;
    let attackLabel = 'none';
    if (attackId !== 'none') {
      const a = ATTACKS.find(x => x.id === attackId);
      if (!a) throw new Error(`unknown attack: ${attackId}`);
      attackLabel = a.label;
      setStatus('dec', `Applying ${attackLabel}…`);
      attacked = await a.fn(coreFrame);
    }

    setStatus('dec', `Decoding ${attacked.width}×${attacked.height}…`);
    const decPerm = permFor(attacked.height, attacked.width);
    const { bits: recCodeword, ms } = await decode(attacked, decPerm);   // 1024 bits
    // Note: we deliberately do NOT releaseSession() here. The decoder worker
    // stays alive across multiple attacks so iOS Safari doesn't overlap two
    // WASM heaps during the ~100ms-1s page-reclaim window. ensureMode() will
    // tear it down cleanly if the user switches back to encoding.

    // Stage 1 — RS(128, 100) decode → 796 wire bits, fixing up to 14 byte errors.
    const eccRes = eccDecode(recCodeword);
    const recWire = eccRes.bits;
    const eccErrors = eccRes.errors;
    const eccOk = eccRes.ok;

    const { syndrome: recSyndrome, sig: recSig, pk: recPk } = unpackWirePayload(recWire);

    // Stage 2 — Slepian-Wolf: recompute pHash on the attacked image and use
    // BCH(127, t=4) with the received syndrome to recover H exactly,
    // correcting up to 7 bits of pHash drift.
    const H_local = canonicalizeH(phash128(attacked));
    const vGuess = bytesToBits(H_local).slice(0, 127);
    const bch = bchDecode(recSyndrome, vGuess);
    const bchErrors = bch.errors;
    const bchOk = bch.ok;

    // Reconstruct 16-byte H from the 127-bit corrected vector (bit 127 = 0).
    const recHBits = new Uint8Array(128);
    recHBits.set(bch.bits, 0);
    const recH = bitsToBytes(recHBits);

    let sigOk = false;
    if (bchOk) {
      try { sigOk = sodium.crypto_sign_verify_detached(recSig, recH, recPk); }
      catch (_) { sigOk = false; }
    }

    // Three bit-accuracy levels:
    //   acc         — 896-bit "logical" payload (H | sig | pk) after BCH + RS
    //   wireAcc     — 817-bit wire (syndrome | sig | pk) after RS
    //   codewordAcc — 1024-bit raw channel
    const knownWire     = (src === 'last') ? lastPayloadBits  : null;
    const knownCodeword = (src === 'last') ? lastCodewordBits : null;
    const knownLogical  = (src === 'last' && lastLogicalParts)
      ? packPayload(lastLogicalParts.H, lastLogicalParts.sig, lastLogicalParts.pk)
      : null;
    const recLogical = packPayload(recH, recSig, recPk);
    const acc         = knownLogical  ? bitAccuracy(recLogical, knownLogical) : null;
    const wireAcc     = knownWire     ? bitAccuracy(recWire, knownWire)       : null;
    const codewordAcc = knownCodeword ? bitAccuracy(recCodeword, knownCodeword) : null;

    const accEl = $('d-acc');
    if (acc != null) {
      accEl.textContent = acc.toFixed(4);
      accEl.classList.toggle('is-ok', acc >= 0.95);
      accEl.classList.toggle('is-bad', acc < 0.80);
    } else {
      accEl.textContent = '—';
      accEl.classList.remove('is-ok', 'is-bad');
    }
    // Detect whether this container was encoded in custom-text mode. For our
    // own last container we know exactly; for uploads we heuristically check
    // whether the recovered sig+pk bytes look like printable UTF-8 text.
    const recTextBytes = new Uint8Array([...recSig, ...recPk]);
    const isLikelyText = (() => {
      let printable = 0, total = 0;
      for (const b of recTextBytes) {
        if (b === 0) continue;  // NUL padding, doesn't count for/against
        total++;
        if ((b >= 0x20 && b <= 0x7E) || b === 0x09 || b === 0x0A) printable++;
      }
      return total > 0 && printable / total > 0.85;
    })();
    const isCustomMode = (src === 'last' && lastSource)
      ? lastSource === 'custom'
      : isLikelyText;

    const sigEl = $('d-sig');
    const sigLblEl = $('d-sig-lbl');
    if (isCustomMode) {
      // Replace the "signature" stat with a "message" stat. ✓ iff text round-
      // tripped exactly (known reference); otherwise just ✓ to indicate "text
      // mode detected — see message line below".
      sigLblEl.textContent = 'message';
      let messageOk = true;
      if (src === 'last' && lastLogicalParts) {
        const refTextBytes = new Uint8Array([
          ...lastLogicalParts.sig, ...lastLogicalParts.pk,
        ]);
        for (let i = 0; i < refTextBytes.length; i++) {
          if (refTextBytes[i] !== recTextBytes[i]) { messageOk = false; break; }
        }
      }
      sigEl.textContent = messageOk ? '✓' : '✗';
      sigEl.classList.toggle('is-ok', messageOk);
      sigEl.classList.toggle('is-bad', !messageOk);
    } else {
      sigLblEl.textContent = 'signature';
      sigEl.textContent = sigOk ? '✓' : '✗';
      sigEl.classList.toggle('is-ok', sigOk);
      sigEl.classList.toggle('is-bad', !sigOk);
    }

    // When a reference is known (decoding our own last container), compute the
    // ACTUAL byte- and bit-error counts so the stat tiles can show real numbers
    // even when ECC/BCH give up. Without a reference we can't know.
    let trueByteErrors = null;
    if (knownCodeword) {
      const recBytes = bitsToBytes(recCodeword);
      const knownBytes = bitsToBytes(knownCodeword);
      trueByteErrors = 0;
      for (let i = 0; i < recBytes.length; i++) if (recBytes[i] !== knownBytes[i]) trueByteErrors++;
    }
    let truePhashDrift = null;
    if (lastH) {
      const knownBits = bytesToBits(lastH).slice(0, 127);
      const localBits = bytesToBits(H_local).slice(0, 127);
      truePhashDrift = 0;
      for (let i = 0; i < 127; i++) if (knownBits[i] !== localBits[i]) truePhashDrift++;
    }

    // Show actual error counts (not the decoder's success/failure flag), colored
    // by whether the count fits in the layer's correction budget. When ECC/BCH
    // give up, the actual count is more useful than the binary "fail" — and
    // BCH can spuriously fail when RS damage corrupted the syndrome input even
    // though the underlying pHash drift was within budget.
    const eccEl = $('d-ecc');
    if (eccEl) {
      const n = trueByteErrors != null ? trueByteErrors : eccErrors;
      eccEl.textContent = n === 0 ? 'clean' : `${n} / ${T_BYTES}`;
      eccEl.classList.toggle('is-ok', n >= 0 && n <= T_BYTES);
      eccEl.classList.toggle('is-bad', n > T_BYTES);
    }
    const bchEl = $('d-bch');
    if (bchEl) {
      const n = truePhashDrift != null ? truePhashDrift : (bchOk ? bchErrors : null);
      if (n != null) {
        bchEl.textContent = n === 0 ? 'clean' : `${n} / ${BCH_T}`;
        bchEl.classList.toggle('is-ok', n <= BCH_T);
        bchEl.classList.toggle('is-bad', n > BCH_T);
      } else {
        bchEl.textContent = '> 7';
        bchEl.classList.toggle('is-ok', false);
        bchEl.classList.toggle('is-bad', true);
      }
    }
    $('d-ms').textContent = `${ms.toFixed(0)} ms`;

    const eccTag = eccOk
      ? (eccErrors === 0 ? 'clean' : `${eccErrors}/${T_BYTES} fixed`)
      : (trueByteErrors != null ? `${trueByteErrors} byte errors (need ≤${T_BYTES})` : 'uncorrectable');
    let bchTag;
    if (bchOk) {
      bchTag = bchErrors === 0 ? 'clean' : `${bchErrors}/${BCH_T} fixed`;
    } else if (truePhashDrift != null && truePhashDrift <= BCH_T) {
      // Surprising case: pHash drift fits in budget but BCH still failed. This
      // happens when RS damage corrupted the syndrome — BCH gets garbage input
      // and can't recover even though the underlying drift was tiny.
      bchTag = `${truePhashDrift} bit drift (would fit; syndrome corrupted by RS-unrecoverable channel damage)`;
    } else if (truePhashDrift != null) {
      bchTag = `${truePhashDrift} bit drift (need ≤${BCH_T})`;
    } else {
      bchTag = 'drift>7';
    }
    const verifyTag = isCustomMode ? '' : `sig: ${sigOk ? 'yes' : 'no'}  ·  `;
    const accLine = acc != null
      ? `logical (H|sig|pk, 896b) acc: ${acc.toFixed(4)}  ·  wire (796b) acc: ${wireAcc.toFixed(4)}  ·  codeword (1024b) acc: ${codewordAcc.toFixed(4)}\n${verifyTag}RS ecc: ${eccTag}  ·  BCH pHash: ${bchTag}`
      : `acc: — (no reference)  ·  ${verifyTag}RS ecc: ${eccTag}  ·  BCH pHash: ${bchTag}`;
    const bitsHtml = chunkBitsHTML(recCodeword, knownCodeword);
    const recSynStr = (() => {
      const g = [];
      for (let i = 0; i < BCH_T; i++) g.push(Array.from(recSyndrome.slice(i * 7, (i + 1) * 7)).join(''));
      return g.join(' ');
    })();
    const recText = bytesToCustomText(recTextBytes);
    const msgEl = $('dec-message');
    const detailsEl = $('dec-details');
    if (isCustomMode) {
      msgEl.innerHTML =
        `<span class="ih-message__label">recovered message</span>` +
        `<span class="ih-message__text">${escapeHtml(recText)}</span>`;
      msgEl.classList.remove('is-hidden');
      detailsEl.classList.add('is-hidden');
    } else {
      msgEl.classList.add('is-hidden');
      detailsEl.classList.remove('is-hidden');
      $('dec-output').innerHTML =
        `image:    ${original.width}×${original.height} → decoded region ${attacked.width}×${attacked.height}\n` +
        `attack:   ${attackLabel}\n` +
        `${accLine}\n` +
        `\n` +
        `H local  : ${hex(H_local)}  (pHash of attacked image, before BCH correction)\n` +
        `H recov  : ${hex(recH)}  (after BCH correction of ${bchErrors >= 0 ? bchErrors : '?'} bit drift)\n` +
        `BCH syn  : ${recSynStr}  (49 bits, recovered from codeword)\n` +
        `sig      : ${hex(recSig)}\n` +
        `pk       : ${hex(recPk)}\n` +
        `\n` +
        `codeword (1024 bits, diff vs encoded):\n${bitsHtml}`;
    }

    $('dec-results').classList.remove('is-hidden');
    setStatus('dec', `Done · ${ms.toFixed(0)} ms.`);
  } catch (e) {
    setStatus('dec', `Decode failed: ${e.message}`, true);
  } finally {
    dec.busy = false; refreshDecRun();
  }
}

// --- bootstrap: kick off init() now that every `let` is initialized ---
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else { init(); }
