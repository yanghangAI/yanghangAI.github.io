import test from 'node:test';
import assert from 'node:assert/strict';
import { computeCrop, splitTrim, pasteBack } from '../../assets/imagehide/trim.js';

test('computeCrop: already-aligned dims pass through unchanged', () => {
  const c = computeCrop(1024, 1024);
  assert.deepEqual(c, { cropH: 1024, cropW: 1024, top: 0, left: 0,
                        trimmedTop: 0, trimmedBottom: 0,
                        trimmedLeft: 0, trimmedRight: 0 });
});

test('computeCrop: 1920x1080 trims to 1920x1024 (center)', () => {
  const c = computeCrop(1080, 1920);
  assert.equal(c.cropH, 1024);   // 1080 - (1080 % 64) = 1080 - 56 = 1024
  assert.equal(c.cropW, 1920);   // 1920 % 64 = 0
  assert.equal(c.top, 28);       // 56 / 2 = 28
  assert.equal(c.left, 0);
  assert.equal(c.trimmedTop, 28);
  assert.equal(c.trimmedBottom, 28);
});

test('computeCrop: odd-trim splits with extra pixel on the bottom/right', () => {
  // 1023 - (1023 % 64) = 1023 - 63 = 960; trim = 63; split = 31 top / 32 bottom
  const c = computeCrop(1023, 1023);
  assert.equal(c.cropH, 960);
  assert.equal(c.trimmedTop, 31);
  assert.equal(c.trimmedBottom, 32);
});

test('splitTrim then pasteBack reconstructs the original RGBA bytes', () => {
  // Construct a 128x128 RGBA ImageData-like object with a deterministic pattern.
  const H = 128, W = 128;
  const data = new Uint8ClampedArray(H * W * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = (i / 4) % 256;
    data[i + 1] = ((i / 4) >> 8) % 256;
    data[i + 2] = ((i / 4) >> 4) % 256;
    data[i + 3] = 255;
  }
  const original = { data, width: W, height: H };
  // computeCrop says 128 → 128 (already aligned), but force a synthetic crop
  // by passing a smaller (cropH, cropW) to splitTrim directly.
  const crop = { cropH: 64, cropW: 64, top: 32, left: 32,
                 trimmedTop: 32, trimmedBottom: 32,
                 trimmedLeft: 32, trimmedRight: 32 };
  const { core, strips } = splitTrim(original, crop);
  assert.equal(core.width, 64);
  assert.equal(core.height, 64);
  const recon = pasteBack(core, strips, crop, W, H);
  assert.equal(recon.width, W);
  assert.equal(recon.height, H);
  assert.deepEqual(recon.data, original.data);
});
