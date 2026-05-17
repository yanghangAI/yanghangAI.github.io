/**
 * Channel attacks for the imagehide demo. All operate on ImageData
 * `{ data: Uint8ClampedArray, width, height }`. RGBA in/out.
 *
 * Attacks:
 *   identity            no-op
 *   jpeg(q)             single JPEG round-trip at quality q (0–100)
 *   jpegChain(q1, q2)   two JPEG passes, q1 first then q2
 *   resize(scale)       bilinear down to scale×scale, then up to original
 *   chain(scale, q)     platform pipeline: resize then JPEG
 *
 * Memory note: every OffscreenCanvas allocated below is freed by setting its
 * width/height to 0 after use, and every ImageBitmap is .close()'d. iOS Safari
 * holds GPU memory for these and the JS GC alone doesn't reliably release it
 * between attack rows — the explicit cleanup is what keeps the tab alive.
 */

function freeCanvas(c) {
  if (c) { c.width = 0; c.height = 0; }
}

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

export async function identity(img) {
  return new ImageData(new Uint8ClampedArray(img.data), img.width, img.height);
}

export async function jpeg(img, q) {
  const blob = await imageDataToBlob(img, 'image/jpeg', q / 100);
  return blobToImageData(blob);
}

export async function jpegChain(img, q1, q2) {
  const once = await jpeg(img, q1);
  return jpeg(once, q2);
}

export async function resize(img, scale) {
  const W = img.width, H = img.height;
  const sw = Math.max(1, Math.round(W * scale));
  const sh = Math.max(1, Math.round(H * scale));

  const srcCanvas = new OffscreenCanvas(W, H);
  srcCanvas.getContext('2d').putImageData(
    new ImageData(img.data, W, H), 0, 0);

  const small = new OffscreenCanvas(sw, sh);
  const sctx = small.getContext('2d');
  sctx.imageSmoothingEnabled = true;
  sctx.imageSmoothingQuality = 'medium';   // 'medium' approximates bilinear
  sctx.drawImage(srcCanvas, 0, 0, sw, sh);
  freeCanvas(srcCanvas);

  const big = new OffscreenCanvas(W, H);
  const bctx = big.getContext('2d');
  bctx.imageSmoothingEnabled = true;
  bctx.imageSmoothingQuality = 'medium';
  bctx.drawImage(small, 0, 0, W, H);
  freeCanvas(small);

  const out = bctx.getImageData(0, 0, W, H);
  freeCanvas(big);
  return out;
}

export async function chain(img, scale, q) {
  const resized = await resize(img, scale);
  return jpeg(resized, q);
}

// Catalog used by the UI. Keep keys stable — they appear as row IDs.
export const ATTACKS = [
  { id: 'identity',            label: 'identity',            fn: (img) => identity(img) },
  { id: 'jpeg_q80',            label: 'JPEG q=80',           fn: (img) => jpeg(img, 80) },
  { id: 'jpeg_q60',            label: 'JPEG q=60',           fn: (img) => jpeg(img, 60) },
  { id: 'jpeg_q40',            label: 'JPEG q=40',           fn: (img) => jpeg(img, 40) },
  { id: 'jpeg_chain_60_40',    label: 'JPEG 60 → 40',        fn: (img) => jpegChain(img, 60, 40) },
  { id: 'resize_075',          label: 'Resize ↓0.75↑',       fn: (img) => resize(img, 0.75) },
  { id: 'resize_050',          label: 'Resize ↓0.5↑',        fn: (img) => resize(img, 0.5) },
  { id: 'chain_insta',         label: 'Instagram (↓0.5↑ + JPEG q=85)',     fn: (img) => chain(img, 0.5, 85) },
  { id: 'chain_x',             label: 'X / Twitter (↓0.4↑ + JPEG q=75)',   fn: (img) => chain(img, 0.4, 75) },
  { id: 'chain_whatsapp_std',  label: 'WhatsApp (↓0.5↑ + JPEG q=70)',      fn: (img) => chain(img, 0.5, 70) },
  { id: 'chain_wechat',        label: 'WeChat (↓0.3↑ + JPEG q=50)',        fn: (img) => chain(img, 0.3, 50) },
];
