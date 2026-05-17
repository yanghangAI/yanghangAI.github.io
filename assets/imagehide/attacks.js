/**
 * Channel attacks for the imagehide demo.
 *
 * I/O convention: every attack consumes a `Frame`
 *   { dataF32: Float32Array,   // length 3*H*W, CHW layout, RGB in [-1, 1]
 *     width:   number,
 *     height:  number }
 * and returns an `ImageData` (uint8 RGBA). The asymmetry is intentional —
 * the model's encoder output is float32 with sub-uint8 residual that we want
 * to preserve through the resize step, but the final JPEG codec and the
 * decoder both consume uint8 anyway, so we quantize only at the canvas
 * boundary. This matches the eval pipeline (`infra/exp0_inn_eval.py`) which
 * keeps the container in float32 between encoder and attack.
 *
 * Resize is implemented in pure JS (not via Canvas2D's `drawImage`) so we
 * can match PyTorch's `F.interpolate(mode='bilinear', align_corners=False)`
 * exactly — including its no-anti-aliasing-on-downsample quirk that the
 * model was trained against. Canvas's `imageSmoothingQuality='medium'`
 * applied a Mitchell-Netravali low-pass that destroyed the high-frequency
 * band the watermark encodes in, dropping `chain_wechat` bit-acc to ~0.80
 * (vs ~0.99 in eval).
 *
 * Memory note: every OffscreenCanvas allocated below is freed by setting
 * its width/height to 0 after use, and every ImageBitmap is .close()'d.
 */

function freeCanvas(c) {
  if (c) { c.width = 0; c.height = 0; }
}

// ---------- Frame ↔ ImageData ----------

export function imageDataToFrame(imageData) {
  const { data, width: W, height: H } = imageData;
  const f32 = new Float32Array(3 * H * W);
  const ch1 = H * W, ch2 = 2 * H * W;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pi = y * W + x;
      const i  = pi * 4;
      f32[pi]       = (data[i]     / 127.5) - 1;
      f32[ch1 + pi] = (data[i + 1] / 127.5) - 1;
      f32[ch2 + pi] = (data[i + 2] / 127.5) - 1;
    }
  }
  return { dataF32: f32, width: W, height: H };
}

export function frameToImageData(frame) {
  const { dataF32, width: W, height: H } = frame;
  const u8 = new Uint8ClampedArray(H * W * 4);
  const ch1 = H * W, ch2 = 2 * H * W;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pi = y * W + x;
      const i  = pi * 4;
      u8[i]     = Math.round(Math.max(0, Math.min(255, (dataF32[pi]       + 1) * 127.5)));
      u8[i + 1] = Math.round(Math.max(0, Math.min(255, (dataF32[ch1 + pi] + 1) * 127.5)));
      u8[i + 2] = Math.round(Math.max(0, Math.min(255, (dataF32[ch2 + pi] + 1) * 127.5)));
      u8[i + 3] = 255;
    }
  }
  return new ImageData(u8, W, H);
}

// ---------- Pure-JS bilinear resize on Float32 CHW ----------

// Matches PyTorch's F.interpolate(input, scale_factor=..., mode='bilinear',
// align_corners=False). Half-pixel alignment: source coord is
//   sx = (ox + 0.5) * (srcW/dstW) - 0.5
// and likewise for sy. Out-of-bounds neighbors are clamped (replicate edge).
// No anti-aliasing pre-filter — the model was trained against this exact
// aliasing behavior.
function resizeOneAxisF32(srcF32, srcW, srcH, dstW, dstH) {
  const out = new Float32Array(3 * dstH * dstW);
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;

  // Precompute x neighbor indices and weights once (same for every row).
  const x0arr = new Int32Array(dstW);
  const x1arr = new Int32Array(dstW);
  const fxArr = new Float32Array(dstW);
  for (let ox = 0; ox < dstW; ox++) {
    const sx = (ox + 0.5) * scaleX - 0.5;
    let sx0 = Math.floor(sx);
    let sx1 = sx0 + 1;
    const fx = sx - sx0;
    if (sx0 < 0)        sx0 = 0;
    else if (sx0 > srcW - 1) sx0 = srcW - 1;
    if (sx1 < 0)        sx1 = 0;
    else if (sx1 > srcW - 1) sx1 = srcW - 1;
    x0arr[ox] = sx0;
    x1arr[ox] = sx1;
    fxArr[ox] = fx;
  }

  for (let c = 0; c < 3; c++) {
    const srcCh = c * srcH * srcW;
    const dstCh = c * dstH * dstW;
    for (let oy = 0; oy < dstH; oy++) {
      const sy = (oy + 0.5) * scaleY - 0.5;
      let sy0 = Math.floor(sy);
      let sy1 = sy0 + 1;
      const fy = sy - sy0;
      if (sy0 < 0)             sy0 = 0;
      else if (sy0 > srcH - 1) sy0 = srcH - 1;
      if (sy1 < 0)             sy1 = 0;
      else if (sy1 > srcH - 1) sy1 = srcH - 1;
      const row0 = srcCh + sy0 * srcW;
      const row1 = srcCh + sy1 * srcW;
      const dstRow = dstCh + oy * dstW;
      for (let ox = 0; ox < dstW; ox++) {
        const x0 = x0arr[ox], x1 = x1arr[ox], fx = fxArr[ox];
        const v00 = srcF32[row0 + x0];
        const v01 = srcF32[row0 + x1];
        const v10 = srcF32[row1 + x0];
        const v11 = srcF32[row1 + x1];
        const v0 = v00 + (v01 - v00) * fx;
        const v1 = v10 + (v11 - v10) * fx;
        out[dstRow + ox] = v0 + (v1 - v0) * fy;
      }
    }
  }
  return out;
}

function resizeDownUpF32(frame, scale) {
  const { dataF32, width: W, height: H } = frame;
  const sw = Math.max(1, Math.round(W * scale));
  const sh = Math.max(1, Math.round(H * scale));
  if (sw === W && sh === H) {
    return { dataF32: new Float32Array(dataF32), width: W, height: H };
  }
  const down = resizeOneAxisF32(dataF32, W, H, sw, sh);
  const up   = resizeOneAxisF32(down, sw, sh, W, H);
  return { dataF32: up, width: W, height: H };
}

// ---------- canvas helpers (used only for JPEG codec) ----------

async function imageDataToBlob(img, type, quality) {
  const c = new OffscreenCanvas(img.width, img.height);
  c.getContext('2d').putImageData(
    new ImageData(img.data, img.width, img.height), 0, 0);
  const blob = await c.convertToBlob({ type, quality });
  freeCanvas(c);
  return blob;
}

async function blobToImageData(blob) {
  const bitmap = await createImageBitmap(blob);
  const c = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = c.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
  const out = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();
  freeCanvas(c);
  return out;
}

async function jpegRoundTripU8(imgU8, q) {
  const blob = await imageDataToBlob(imgU8, 'image/jpeg', q / 100);
  return blobToImageData(blob);
}

// ---------- attacks ----------

export async function identity(frame) {
  return frameToImageData(frame);
}

export async function jpeg(frame, q) {
  return jpegRoundTripU8(frameToImageData(frame), q);
}

export async function jpegChain(frame, q1, q2) {
  const after1 = await jpegRoundTripU8(frameToImageData(frame), q1);
  return jpegRoundTripU8(after1, q2);
}

export async function resize(frame, scale) {
  return frameToImageData(resizeDownUpF32(frame, scale));
}

export async function chain(frame, scale, q) {
  const resized = resizeDownUpF32(frame, scale);     // float32 → float32
  return jpegRoundTripU8(frameToImageData(resized), q);
}

// Catalog used by the UI. Keep keys stable — they appear as row IDs.
export const ATTACKS = [
  { id: 'identity',            label: 'identity',                            fn: (f) => identity(f) },
  { id: 'jpeg_q80',            label: 'JPEG q=80',                           fn: (f) => jpeg(f, 80) },
  { id: 'jpeg_q60',            label: 'JPEG q=60',                           fn: (f) => jpeg(f, 60) },
  { id: 'jpeg_q40',            label: 'JPEG q=40',                           fn: (f) => jpeg(f, 40) },
  { id: 'jpeg_chain_60_40',    label: 'JPEG 60 → 40',                        fn: (f) => jpegChain(f, 60, 40) },
  { id: 'resize_075',          label: 'Resize ↓0.75↑',                       fn: (f) => resize(f, 0.75) },
  { id: 'resize_050',          label: 'Resize ↓0.5↑',                        fn: (f) => resize(f, 0.5) },
  { id: 'chain_insta',         label: 'Instagram (↓0.5↑ + JPEG q=85)',       fn: (f) => chain(f, 0.5, 85) },
  { id: 'chain_x',             label: 'X / Twitter (↓0.4↑ + JPEG q=75)',     fn: (f) => chain(f, 0.4, 75) },
  { id: 'chain_whatsapp_std',  label: 'WhatsApp (↓0.5↑ + JPEG q=70)',        fn: (f) => chain(f, 0.5, 70) },
  { id: 'chain_wechat',        label: 'WeChat (↓0.3↑ + JPEG q=50)',          fn: (f) => chain(f, 0.3, 50) },
];
