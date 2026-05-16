import test from 'node:test';
import assert from 'node:assert/strict';
import { bitsToBytes, bytesToBits, packPayload, unpackPayload,
         phash128 } from '../../assets/imagehide/payload.js';

test('bitsToBytes / bytesToBits round-trip on aligned length', () => {
  const bits = Uint8Array.from(
    Array.from({ length: 32 }, (_, i) => i % 2));
  const bytes = bitsToBytes(bits);
  assert.equal(bytes.length, 4);
  const back = bytesToBits(bytes);
  assert.deepEqual(Array.from(back), Array.from(bits));
});

test('bitsToBytes: known bit pattern → known bytes (MSB-first per byte)', () => {
  // 1000 0000 → 0x80
  const bits = Uint8Array.from([1, 0, 0, 0, 0, 0, 0, 0]);
  assert.deepEqual(Array.from(bitsToBytes(bits)), [0x80]);
});

test('packPayload returns exactly 896 bits with the three regions', () => {
  const H_bytes = new Uint8Array(16).fill(0xAA);   // 128 bits
  const sig_bytes = new Uint8Array(64).fill(0xBB); // 512 bits
  const pk_bytes = new Uint8Array(32).fill(0xCC);  // 256 bits
  const bits = packPayload(H_bytes, sig_bytes, pk_bytes);
  assert.equal(bits.length, 896);
  // First 128 bits should be the H byte pattern
  assert.deepEqual(Array.from(bits.slice(0, 8)),  [1, 0, 1, 0, 1, 0, 1, 0]);
  // Bits 128..640 are sig (0xBB = 1011 1011)
  assert.deepEqual(Array.from(bits.slice(128, 136)), [1, 0, 1, 1, 1, 0, 1, 1]);
  // Bits 640..896 are pk (0xCC = 1100 1100)
  assert.deepEqual(Array.from(bits.slice(640, 648)), [1, 1, 0, 0, 1, 1, 0, 0]);
});

test('unpackPayload inverts packPayload', () => {
  const H = crypto.getRandomValues(new Uint8Array(16));
  const sig = crypto.getRandomValues(new Uint8Array(64));
  const pk = crypto.getRandomValues(new Uint8Array(32));
  const bits = packPayload(H, sig, pk);
  const u = unpackPayload(bits);
  assert.deepEqual(Array.from(u.H), Array.from(H));
  assert.deepEqual(Array.from(u.sig), Array.from(sig));
  assert.deepEqual(Array.from(u.pk), Array.from(pk));
});

test('phash128 returns 16 bytes (128 bits) and is deterministic', () => {
  const H = 64, W = 64;
  const rgba = new Uint8ClampedArray(H * W * 4);
  for (let i = 0; i < rgba.length; i += 4) {
    rgba[i] = i % 256; rgba[i+1] = (i*3) % 256;
    rgba[i+2] = (i*7) % 256; rgba[i+3] = 255;
  }
  const a = phash128({ data: rgba, width: W, height: H });
  const b = phash128({ data: rgba, width: W, height: H });
  assert.equal(a.length, 16);
  assert.deepEqual(Array.from(a), Array.from(b));
});

test('phash128 changes when image changes substantially', () => {
  const H = 64, W = 64;
  const rgbaA = new Uint8ClampedArray(H * W * 4).fill(0);
  const rgbaB = new Uint8ClampedArray(H * W * 4).fill(255);
  for (let i = 3; i < rgbaA.length; i += 4) { rgbaA[i] = 255; rgbaB[i] = 255; }
  const a = phash128({ data: rgbaA, width: W, height: H });
  const b = phash128({ data: rgbaB, width: W, height: H });
  // At least 32 of 128 bits should differ
  let diff = 0;
  for (let i = 0; i < 16; i++) {
    let x = a[i] ^ b[i];
    while (x) { diff += x & 1; x >>= 1; }
  }
  assert.ok(diff >= 32, `expected ≥32 bit differences, got ${diff}`);
});
