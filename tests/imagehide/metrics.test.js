import test from 'node:test';
import assert from 'node:assert/strict';
import { psnr, ssim } from '../../assets/imagehide/metrics.js';

function makeRgba(w, h, fill) {
  const d = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < d.length; i += 4) {
    d[i] = fill; d[i+1] = fill; d[i+2] = fill; d[i+3] = 255;
  }
  return { data: d, width: w, height: h };
}

test('psnr: identical images → Infinity', () => {
  const a = makeRgba(32, 32, 128);
  const b = makeRgba(32, 32, 128);
  assert.equal(psnr(a, b), Infinity);
});

test('psnr: known MSE → known PSNR', () => {
  // All-zero vs all-1 over RGB (alpha skipped): MSE = 1, PSNR = 20*log10(255) ≈ 48.13
  const a = makeRgba(8, 8, 0);
  const b = makeRgba(8, 8, 1);
  const v = psnr(a, b);
  assert.ok(Math.abs(v - 48.131) < 0.01, `got ${v}`);
});

test('ssim: identical images → 1.0', () => {
  const a = makeRgba(64, 64, 100);
  const b = makeRgba(64, 64, 100);
  const v = ssim(a, b);
  assert.ok(Math.abs(v - 1.0) < 1e-6, `got ${v}`);
});

test('ssim: very different images → near 0 or low', () => {
  const a = makeRgba(64, 64, 0);
  const b = makeRgba(64, 64, 255);
  const v = ssim(a, b);
  assert.ok(v < 0.1, `expected very low SSIM for max contrast, got ${v}`);
});
