// PDQ perceptual hash, ported from Facebook ThreatExchange's reference
// implementation. Output truncated to 128 bits (low-frequency zigzag).
//
// Algorithm (operating on RGBA ImageData of arbitrary size):
//   1. Rec.601 luma (0.299 R + 0.587 G + 0.114 B)
//   2. Jarosz filter: two 1D box-average passes per axis, window size =
//      max(2, max(W, H) / 64). Approximates a Gaussian filter; adaptive
//      window keeps the smoothing scale consistent across input resolutions.
//   3. Decimate to 64x64 via INTER_AREA-style block averaging, computed in
//      O(64^2) via an integral image of the smoothed luminance.
//   4. Separable 64x64 type-II DCT.
//   5. Take top-left 16x16 sub-block (256 lowest-freq coefficients).
//   6. Threshold each at the median of all 256 -> 256 bits (PDQ standard).
//   7. For 128-bit output: take the first 128 bits in zigzag order over the
//      16x16 block (lowest-frequency first).
//
// Empirical drift sweep on 30 COCO val2017 images at 512x512, vs the demo's
// 11-attack catalog: median worst-attack drift = 1/128, p95 = 3/128, max = 3/128.
// BCH(127, 78, t=7) covers 100% of images with ample margin.

// 3-channel luminance pass: RGBA -> Float64 W*H grayscale.
function rgbaToLuma(imageData) {
  const { data, width: W, height: H } = imageData;
  const out = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const p = i * 4;
    out[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  }
  return out;
}

// 1D centered box-average along one axis, edges replicated (clamped).
// O(W*H) using row-wise prefix sums.
function boxAvgHoriz(src, W, H, win) {
  const out = new Float64Array(W * H);
  const half = win >> 1;
  const prefix = new Float64Array(W + 1);
  for (let y = 0; y < H; y++) {
    const off = y * W;
    prefix[0] = 0;
    for (let x = 0; x < W; x++) prefix[x + 1] = prefix[x] + src[off + x];
    for (let x = 0; x < W; x++) {
      let xa = x - half;
      let xb = x - half + win;
      if (xa < 0) xa = 0;
      if (xb > W) xb = W;
      out[off + x] = (prefix[xb] - prefix[xa]) / (xb - xa);
    }
  }
  return out;
}
function boxAvgVert(src, W, H, win) {
  const out = new Float64Array(W * H);
  const half = win >> 1;
  const prefix = new Float64Array(H + 1);
  for (let x = 0; x < W; x++) {
    prefix[0] = 0;
    for (let y = 0; y < H; y++) prefix[y + 1] = prefix[y] + src[y * W + x];
    for (let y = 0; y < H; y++) {
      let ya = y - half;
      let yb = y - half + win;
      if (ya < 0) ya = 0;
      if (yb > H) yb = H;
      out[y * W + x] = (prefix[yb] - prefix[ya]) / (yb - ya);
    }
  }
  return out;
}
// Two passes per axis (total 4) — the standard PDQ Jarosz configuration.
function jaroszFilter(arr, W, H, win) {
  let a = boxAvgHoriz(arr, W, H, win);
  a = boxAvgHoriz(a, W, H, win);
  a = boxAvgVert(a, W, H, win);
  a = boxAvgVert(a, W, H, win);
  return a;
}

// Decimate W x H luma to 64 x 64 via integer-aligned block averaging.
// Uses an integral image so each output pixel is O(1) regardless of source.
function decimateTo64(src, W, H) {
  const intW = W + 1;
  const integral = new Float64Array((H + 1) * intW);
  for (let y = 0; y < H; y++) {
    const ioff = (y + 1) * intW;
    const ooff = y * intW;
    let rowsum = 0;
    for (let x = 0; x < W; x++) {
      rowsum += src[y * W + x];
      integral[ioff + x + 1] = integral[ooff + x + 1] + rowsum;
    }
  }
  const out = new Float64Array(64 * 64);
  for (let oy = 0; oy < 64; oy++) {
    const yA = (oy * H) >> 6;
    let yB = ((oy + 1) * H) >> 6;
    if (yB <= yA) yB = yA + 1;
    const yh = yB - yA;
    for (let ox = 0; ox < 64; ox++) {
      const xA = (ox * W) >> 6;
      let xB = ((ox + 1) * W) >> 6;
      if (xB <= xA) xB = xA + 1;
      const xw = xB - xA;
      const s = integral[yB * intW + xB]
              - integral[yA * intW + xB]
              - integral[yB * intW + xA]
              + integral[yA * intW + xA];
      out[oy * 64 + ox] = s / (xw * yh);
    }
  }
  return out;
}

// Precomputed 64x64 cosine kernel (computed once, cached).
const _N = 64;
let _cos = null;
function _getCosKernel() {
  if (_cos) return _cos;
  const c = new Float64Array(_N * _N);
  for (let u = 0; u < _N; u++) {
    const cu = u === 0 ? Math.SQRT1_2 : 1;
    for (let x = 0; x < _N; x++) {
      c[u * _N + x] = cu * Math.cos(((2 * x + 1) * u * Math.PI) / (2 * _N));
    }
  }
  _cos = c;
  return c;
}

// Separable 64x64 type-II DCT, O(N^3) instead of naive O(N^4).
// Output layout: out[v * 64 + u] = DCT coefficient at frequency (u, v).
// Normalization is omitted (bit thresholds at median are scale-invariant).
function dct64(input) {
  const c = _getCosKernel();
  const tmp = new Float64Array(_N * _N);
  for (let y = 0; y < _N; y++) {
    for (let u = 0; u < _N; u++) {
      let s = 0;
      for (let x = 0; x < _N; x++) {
        s += input[y * _N + x] * c[u * _N + x];
      }
      tmp[y * _N + u] = s;
    }
  }
  const out = new Float64Array(_N * _N);
  for (let u = 0; u < _N; u++) {
    for (let v = 0; v < _N; v++) {
      let s = 0;
      for (let y = 0; y < _N; y++) {
        s += tmp[y * _N + u] * c[v * _N + y];
      }
      out[v * _N + u] = s;
    }
  }
  return out;
}

// Zigzag order over a 16x16 grid (anti-diagonal). Returns up to 128 (u, v) tuples.
function _zigzag16_128() {
  const out = [];
  for (let s = 0; s < 31 && out.length < 128; s++) {
    if (s & 1) {
      for (let u = Math.max(0, s - 15); u <= Math.min(s, 15) && out.length < 128; u++) {
        out.push([u, s - u]);
      }
    } else {
      for (let v = Math.max(0, s - 15); v <= Math.min(s, 15) && out.length < 128; v++) {
        out.push([s - v, v]);
      }
    }
  }
  return out;
}
const _ZZ16_128 = _zigzag16_128();

// Compute the PDQ-style 128-bit perceptual hash of an RGBA ImageData.
// Returns a Uint8Array(16) — 128 packed bits, MSB-first within each byte.
export function pdq128(imageData) {
  const { width: W, height: H } = imageData;
  const luma = rgbaToLuma(imageData);
  const win = Math.max(2, Math.floor(Math.max(W, H) / 64));
  const smoothed = jaroszFilter(luma, W, H, win);
  const small = decimateTo64(smoothed, W, H);
  const dct = dct64(small);

  // Top-left 16x16 sub-block (256 lowest-freq coefficients).
  const block16 = new Float64Array(256);
  for (let v = 0; v < 16; v++) {
    for (let u = 0; u < 16; u++) {
      block16[v * 16 + u] = dct[v * _N + u];
    }
  }
  // Median threshold (PDQ standard).
  const sorted = Float64Array.from(block16);
  sorted.sort();
  const median = (sorted[127] + sorted[128]) / 2;

  // Take first 128 bits in zigzag order over 16x16.
  const bits = new Uint8Array(128);
  for (let i = 0; i < 128; i++) {
    const [u, v] = _ZZ16_128[i];
    bits[i] = block16[v * 16 + u] > median ? 1 : 0;
  }
  // Pack to 16 bytes MSB-first.
  const out = new Uint8Array(16);
  for (let i = 0; i < 16; i++) {
    let b = 0;
    for (let j = 0; j < 8; j++) b = (b << 1) | bits[i * 8 + j];
    out[i] = b;
  }
  return out;
}
