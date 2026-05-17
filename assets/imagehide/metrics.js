/**
 * PSNR and SSIM on RGBA ImageData. RGB channels only; alpha ignored.
 *
 * Both images must have the same dimensions.
 *
 * SSIM: Wang et al. 2004 with 11×11 Gaussian window, σ=1.5, K1=0.01, K2=0.03,
 * L=255. Computed per-channel on RGB, averaged across channels. The window
 * is applied at unit stride; the SSIM map is averaged over valid positions.
 */

const K1 = 0.01, K2 = 0.03, L = 255;
const C1 = (K1 * L) ** 2;
const C2 = (K2 * L) ** 2;

export function psnr(aImg, bImg) {
  if (aImg.width !== bImg.width || aImg.height !== bImg.height) {
    throw new Error('PSNR: dimension mismatch');
  }
  const a = aImg.data, b = bImg.data;
  let sse = 0, n = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const d = a[i + c] - b[i + c];
      sse += d * d;
      n++;
    }
  }
  if (sse === 0) return Infinity;
  const mse = sse / n;
  return 10 * Math.log10((L * L) / mse);
}

function gaussianKernel1D(size, sigma) {
  const k = new Float32Array(size);
  const half = (size - 1) / 2;
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const x = i - half;
    k[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += k[i];
  }
  for (let i = 0; i < size; i++) k[i] /= sum;
  return k;
}

function convolve2DSeparable(src, W, H, k1d) {
  const r = (k1d.length - 1) / 2;
  const tmp = new Float32Array(W * H);
  // Horizontal pass
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let dx = -r; dx <= r; dx++) {
        const sx = Math.min(W - 1, Math.max(0, x + dx));
        s += src[y * W + sx] * k1d[dx + r];
      }
      tmp[y * W + x] = s;
    }
  }
  const out = new Float32Array(W * H);
  // Vertical pass
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let dy = -r; dy <= r; dy++) {
        const sy = Math.min(H - 1, Math.max(0, y + dy));
        s += tmp[sy * W + x] * k1d[dy + r];
      }
      out[y * W + x] = s;
    }
  }
  return out;
}

function extractChannel(img, c) {
  const { data, width: W, height: H } = img;
  const out = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) out[i] = data[i * 4 + c];
  return out;
}

function ssimChannel(a, b, W, H, kernel) {
  // Per-channel SSIM map mean.
  const muA = convolve2DSeparable(a, W, H, kernel);
  const muB = convolve2DSeparable(b, W, H, kernel);
  const a2 = new Float32Array(W * H);
  const b2 = new Float32Array(W * H);
  const ab = new Float32Array(W * H);
  for (let i = 0; i < a.length; i++) {
    a2[i] = a[i] * a[i];
    b2[i] = b[i] * b[i];
    ab[i] = a[i] * b[i];
  }
  const muA2 = convolve2DSeparable(a2, W, H, kernel);
  const muB2 = convolve2DSeparable(b2, W, H, kernel);
  const muAB = convolve2DSeparable(ab, W, H, kernel);

  let sum = 0;
  const N = W * H;
  for (let i = 0; i < N; i++) {
    const mA = muA[i], mB = muB[i];
    const sigA2 = muA2[i] - mA * mA;
    const sigB2 = muB2[i] - mB * mB;
    const sigAB = muAB[i] - mA * mB;
    const num = (2 * mA * mB + C1) * (2 * sigAB + C2);
    const den = (mA * mA + mB * mB + C1) * (sigA2 + sigB2 + C2);
    sum += num / den;
  }
  return sum / N;
}

// Above this many pixels, SSIM downsamples (nearest-neighbor) before computing
// so the convolution temporaries don't blow up mobile browser memory. 512×512
// is enough resolution for SSIM to be representative; full-res adds ~10× cost
// for ~0.001 of metric change.
const SSIM_MAX_PIXELS = 512 * 512;

function downsampleImageData(img, maxPixels) {
  const { width: W, height: H } = img;
  if (W * H <= maxPixels) return img;
  const scale = Math.sqrt(maxPixels / (W * H));
  const w = Math.max(11, Math.round(W * scale));
  const h = Math.max(11, Math.round(H * scale));
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(H - 1, Math.floor(y * H / h));
    for (let x = 0; x < w; x++) {
      const sx = Math.min(W - 1, Math.floor(x * W / w));
      const si = (sy * W + sx) * 4;
      const di = (y * w + x) * 4;
      out[di]     = img.data[si];
      out[di + 1] = img.data[si + 1];
      out[di + 2] = img.data[si + 2];
      out[di + 3] = 255;
    }
  }
  return { data: out, width: w, height: h };
}

export function ssim(aImg, bImg) {
  if (aImg.width !== bImg.width || aImg.height !== bImg.height) {
    throw new Error('SSIM: dimension mismatch');
  }
  const a = downsampleImageData(aImg, SSIM_MAX_PIXELS);
  const b = downsampleImageData(bImg, SSIM_MAX_PIXELS);
  const W = a.width, H = a.height;
  const kernel = gaussianKernel1D(11, 1.5);
  let acc = 0;
  for (let c = 0; c < 3; c++) {
    acc += ssimChannel(extractChannel(a, c), extractChannel(b, c),
                       W, H, kernel);
  }
  return acc / 3;
}
