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

// Standard type-II 2D DCT on an N×N block. Naive O(n^4); fine for N ≤ 16.
function dct2NxN(block, N) {
  const out = new Float64Array(N * N);
  // Precompute cosines.
  const cos = new Float64Array(N * N);
  for (let u = 0; u < N; u++) {
    for (let x = 0; x < N; x++) {
      cos[u * N + x] = Math.cos(((2 * x + 1) * u * Math.PI) / (2 * N));
    }
  }
  for (let u = 0; u < N; u++) {
    for (let v = 0; v < N; v++) {
      let s = 0;
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          s += block[y * N + x] * cos[u * N + x] * cos[v * N + y];
        }
      }
      const cu = u === 0 ? Math.SQRT1_2 : 1;
      const cv = v === 0 ? Math.SQRT1_2 : 1;
      out[u * N + v] = (cu * cv * s) * 2 / N;
    }
  }
  return out;
}

// Zigzag order over a 12×12 grid: anti-diagonal traversal. Used to pick the
// 128 lowest-frequency coefficients (after dropping DC) from a larger DCT.
function zigzag12() {
  const out = [];
  for (let s = 0; s < 12 + 12 - 1; s++) {
    if (s & 1) {
      for (let u = Math.max(0, s - 11); u <= Math.min(s, 11); u++) {
        out.push([u, s - u]);
      }
    } else {
      for (let v = Math.max(0, s - 11); v <= Math.min(s, 11); v++) {
        out.push([s - v, v]);
      }
    }
  }
  return out;  // length 144
}

// Real 128-bit DCT pHash — 128 *unique* bits.
//
// Pipeline:
//   32×32 grayscale (Rec.601 luminance, nearest-neighbor downscale)
//   → 2×2 mean-pool to 16×16
//   → 16×16 type-II DCT (256 coeffs)
//   → take the top-left 12×12 sub-block (144 lowest-freq coeffs)
//   → drop DC, threshold the remaining 143 at their median
//   → traverse 12×12 in zigzag order, pick the first 128 non-DC coeffs
//     → 128 independent bits (one per coefficient)
//
// Previous version computed the standard 64-bit 8×8 DCT pHash and duplicated
// 64 bits to fill 128 — burned half the H slot's collision resistance and
// halved BCH-syndrome efficiency. This replacement actually carries 128 bits
// of entropy. Coefficients are in the same frequency range as the prior 8×8
// pHash (max ~11/32 cycles/source-pixel vs prior 7/32) so robustness to JPEG /
// resize attacks is comparable.
export function phash128(imageData) {
  const gray32 = rgbaToGrayResized(imageData, 32);

  // 2×2 mean-pool to 16×16.
  const block16 = new Float64Array(16 * 16);
  for (let by = 0; by < 16; by++) {
    for (let bx = 0; bx < 16; bx++) {
      let s = 0;
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          s += gray32[(by * 2 + dy) * 32 + (bx * 2 + dx)];
        }
      }
      block16[by * 16 + bx] = s / 4;
    }
  }

  const dct = dct2NxN(block16, 16);

  // Collect the 143 non-DC coefficients from the top-left 12×12 sub-block.
  const coeffs = new Float64Array(143);
  let k = 0;
  for (let u = 0; u < 12; u++) {
    for (let v = 0; v < 12; v++) {
      if (u === 0 && v === 0) continue;
      coeffs[k++] = dct[u * 16 + v];
    }
  }
  const sorted = Array.from(coeffs).sort((a, b) => a - b);
  const median = sorted[71];  // 143 / 2 ≈ 71

  // Pick the first 128 non-DC coefficients in zigzag order.
  const bits = new Uint8Array(128);
  const zz = zigzag12();
  let idx = 0;
  for (const [u, v] of zz) {
    if (u === 0 && v === 0) continue;
    if (idx >= 128) break;
    bits[idx++] = dct[u * 16 + v] > median ? 1 : 0;
  }
  return bitsToBytes(bits);
}
