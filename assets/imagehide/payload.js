/**
 * Payload helpers for the imagehide demo.
 *
 * The 896-bit MWIP payload is [H(128) | sig(512) | pk(256)].
 *
 * Bit/byte conventions: MSB-first within each byte (matches Ed25519 and the
 * model's bit-encoding adapter).
 *
 * pHash: 32×32 grayscale → 8×8 DCT (top-left low-frequency block) → mean
 * threshold → 64 bits → repeat-twice padding to 128 bits to match the H slot.
 * Honest placeholder; the production MWIP H function will replace this.
 */

export const N_BITS = 896;
export const N_H = 128;
export const N_SIG = 512;
export const N_PK = 256;

export function bitsToBytes(bits) {
  if (bits.length % 8 !== 0) throw new Error('bits.length must be a multiple of 8');
  const out = new Uint8Array(bits.length / 8);
  for (let i = 0; i < bits.length; i++) {
    if (bits[i]) out[i >> 3] |= 1 << (7 - (i & 7));
  }
  return out;
}

export function bytesToBits(bytes) {
  const out = new Uint8Array(bytes.length * 8);
  for (let i = 0; i < bytes.length; i++) {
    for (let b = 0; b < 8; b++) {
      out[i * 8 + b] = (bytes[i] >> (7 - b)) & 1;
    }
  }
  return out;
}

export function packPayload(H_bytes, sig_bytes, pk_bytes) {
  if (H_bytes.length !== N_H / 8)   throw new Error('H must be 16 bytes');
  if (sig_bytes.length !== N_SIG / 8) throw new Error('sig must be 64 bytes');
  if (pk_bytes.length !== N_PK / 8)   throw new Error('pk must be 32 bytes');
  const all = new Uint8Array(N_BITS / 8);
  all.set(H_bytes, 0);
  all.set(sig_bytes, N_H / 8);
  all.set(pk_bytes, (N_H + N_SIG) / 8);
  return bytesToBits(all);
}

export function unpackPayload(bits) {
  if (bits.length !== N_BITS) throw new Error(`bits must be ${N_BITS} long`);
  const bytes = bitsToBytes(bits);
  return {
    H:   bytes.slice(0, N_H / 8),
    sig: bytes.slice(N_H / 8, (N_H + N_SIG) / 8),
    pk:  bytes.slice((N_H + N_SIG) / 8),
  };
}

export function bitAccuracy(a, b) {
  if (a.length !== b.length) throw new Error('length mismatch');
  let same = 0;
  for (let i = 0; i < a.length; i++) if (a[i] === b[i]) same++;
  return same / a.length;
}

// ---------- pHash ----------

function rgbaToGrayResized(imageData, target) {
  // Nearest-neighbor downscale to target×target, with luminance conversion.
  const { data, width: W, height: H } = imageData;
  const out = new Float64Array(target * target);
  for (let y = 0; y < target; y++) {
    const sy = Math.floor(y * H / target);
    for (let x = 0; x < target; x++) {
      const sx = Math.floor(x * W / target);
      const i = (sy * W + sx) * 4;
      out[y * target + x] = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
    }
  }
  return out;
}

function dct2_8x8(block) {
  // Standard type-II 2D DCT on an 8×8 block. Naive O(n^4) is fine for this size.
  const N = 8;
  const out = new Float64Array(N * N);
  for (let u = 0; u < N; u++) {
    for (let v = 0; v < N; v++) {
      let s = 0;
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          s += block[y * N + x]
               * Math.cos(((2 * x + 1) * u * Math.PI) / (2 * N))
               * Math.cos(((2 * y + 1) * v * Math.PI) / (2 * N));
        }
      }
      const cu = u === 0 ? Math.SQRT1_2 : 1;
      const cv = v === 0 ? Math.SQRT1_2 : 1;
      out[u * N + v] = (cu * cv * s) / 4;
    }
  }
  return out;
}

export function phash128(imageData) {
  // 32×32 gray → take 8×8 average pool to feed DCT8 → mean threshold → 64 bits
  // → repeat twice to fill 128.
  const gray32 = rgbaToGrayResized(imageData, 32);
  const block8 = new Float64Array(64);
  for (let by = 0; by < 8; by++) {
    for (let bx = 0; bx < 8; bx++) {
      let s = 0;
      for (let dy = 0; dy < 4; dy++) {
        for (let dx = 0; dx < 4; dx++) {
          s += gray32[(by * 4 + dy) * 32 + (bx * 4 + dx)];
        }
      }
      block8[by * 8 + bx] = s / 16;
    }
  }
  const dct = dct2_8x8(block8);
  // Use the 8×8 block excluding DC (top-left). Threshold on median of remaining 63.
  const lowfreq = Array.from(dct.slice(1));
  const sorted = [...lowfreq].sort((a, b) => a - b);
  const median = sorted[31];
  const bits64 = new Uint8Array(64);
  bits64[0] = 1;  // DC bit fixed; not informative
  for (let i = 1; i < 64; i++) bits64[i] = lowfreq[i - 1] > median ? 1 : 0;
  // Pack 64 bits → 8 bytes, then duplicate to fill 16 bytes (128 bits)
  const eight = bitsToBytes(bits64);
  const sixteen = new Uint8Array(16);
  sixteen.set(eight, 0);
  sixteen.set(eight, 8);
  return sixteen;
}
