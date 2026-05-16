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
  const k = new Float64Array(size);
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
  const tmp = new Float64Array(W * H);
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
  const out = new Float64Array(W * H);
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
  const out = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) out[i] = data[i * 4 + c];
  return out;
}

function ssimChannel(a, b, W, H, kernel) {
  // Per-channel SSIM map mean.
  const muA = convolve2DSeparable(a, W, H, kernel);
  const muB = convolve2DSeparable(b, W, H, kernel);
  const a2 = new Float64Array(W * H);
  const b2 = new Float64Array(W * H);
  const ab = new Float64Array(W * H);
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

export function ssim(aImg, bImg) {
  if (aImg.width !== bImg.width || aImg.height !== bImg.height) {
    throw new Error('SSIM: dimension mismatch');
  }
  const W = aImg.width, H = aImg.height;
  const kernel = gaussianKernel1D(11, 1.5);
  let acc = 0;
  for (let c = 0; c < 3; c++) {
    acc += ssimChannel(extractChannel(aImg, c), extractChannel(bImg, c),
                       W, H, kernel);
  }
  return acc / 3;
}
