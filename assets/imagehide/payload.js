/**
 * Payload helpers for the imagehide demo.
 *
 * The 896-bit MWIP user-level payload is [H(128) | sig(512) | pk(256)]; the
 * Slepian-Wolf wire payload that the model embeds replaces H with a 49-bit
 * BCH syndrome (see bch.js + ecc.js).
 *
 * Bit/byte conventions: MSB-first within each byte (matches Ed25519 and the
 * model's bit-encoding adapter).
 *
 * pHash: full PDQ (Facebook ThreatExchange), implemented in pdq.js, truncated
 * to 128 bits via zigzag over the top-left 16x16 sub-block.
 */

// Top-of-file static import so the browser resolves the dependency before any
// other code in this module runs. Re-exported so callers that need direct PDQ
// access don't have to import the module separately.
import { pdq128 } from './pdq.js';
export { pdq128 };

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

// Standard type-II 2D DCT on an 8×8 block. Used for the low-freq half of phash128.
function dct2_8x8(block) {
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

// 128-bit perceptual hash: full PDQ (low-frequency 128 of the 256-bit PDQ hash).
//
// PDQ is Facebook ThreatExchange's perceptual hash, designed for adversarial
// image-match detection in the wild (used by NCMEC for CSAM matching). On our
// 30-image COCO sweep through the demo's 11-attack catalog, PDQ caps drift at
// 3/128 bits (vs 12/128 for the previous blurred-hybrid), so BCH(127, 78, t=7)
// covers 100% of typical photos with multiple bits of headroom.
//
// Implementation lives in pdq.js. Cost: ~30-60ms per call on a 512x512 input
// (vs ~10ms for the previous hybrid). Negligible next to the model's ~1-5s
// forward pass.
export function phash128(imageData) {
  return pdq128(imageData);
}
