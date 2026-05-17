// Reed-Solomon RS(128, 103) over GF(2^8) with primitive polynomial 0x11D
// (the standard x^8 + x^4 + x^3 + x^2 + 1 used in RS for QR codes, CCSDS, etc).
//
//   Data:    103 bytes  (817 user bits + 7 zero pad)
//   Parity:  25 bytes   (200 redundancy bits)
//   Total:   128 bytes  (1024 codeword bits — matches the 1024-bit INN model)
//
// Corrects up to floor(25/2) = 12 byte errors anywhere in the codeword.
// (Up from 8 in the older RS(128, 112). The savings come from Slepian-Wolf
// pHash compression: we transmit the 49-bit BCH syndrome of pHash instead of
// the 128-bit hash itself, freeing 79 bits we reinvest in ECC parity.)
//
// Wire payload layout (817 bits — exported as PAYLOAD_BITS):
//   [0   .. 49 )  BCH(127, 78, t=7) syndrome of pHash
//   [49  .. 561)  Ed25519 signature (512 bits)
//   [561 ..817 )  Ed25519 public key (256 bits)
//
// Conventions inside the RS code:
//   * Codeword byte layout: code[0..K-1] = data bytes, code[K..N-1] = parity
//   * Polynomial: code[i] = coefficient of x^(N-1-i); code[0] = highest deg
//   * Generator: narrow-sense, g(x) = ∏(x - α^i) for i in [0, NSYM)
//   * Berlekamp-Massey σ stored low-to-high (sigma[0] = constant term = 1)

export const N = 128;
export const K = 103;
export const NSYM = N - K;            // 25 parity bytes; t = NSYM/2 = 12
export const T_BYTES = NSYM >> 1;     // 12
export const PAYLOAD_BITS = 817;
export const CODEWORD_BITS = N * 8;   // 1024
const PAD_BITS = K * 8 - PAYLOAD_BITS; // 7 trailing zero pad bits

const PRIM_POLY = 0x11D;
const EXP = new Uint8Array(512);
const LOG = new Int16Array(256);

(function initGF() {
  for (let i = 0; i < 256; i++) LOG[i] = -1;
  let x = 1;
  for (let i = 0; i < 255; i++) {
    EXP[i] = x;
    LOG[x] = i;
    x <<= 1;
    if (x & 0x100) x ^= PRIM_POLY;
  }
  for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255];
})();

const gMul = (a, b) => (a === 0 || b === 0) ? 0 : EXP[LOG[a] + LOG[b]];
const gInv = (a) => { if (a === 0) throw new Error('gInv(0)'); return EXP[255 - LOG[a]]; };

// Generator g(x) = (x - α^0)(x - α^1)...(x - α^(NSYM-1))
// Coefficients laid out highest-degree first: GEN[0] = leading coef = 1.
const GEN = (function buildGen() {
  let g = new Uint8Array([1]);
  for (let i = 0; i < NSYM; i++) {
    const ai = EXP[i];
    const next = new Uint8Array(g.length + 1);
    for (let j = 0; j < g.length; j++) {
      next[j] ^= g[j];
      next[j + 1] ^= gMul(g[j], ai);
    }
    g = next;
  }
  return g;
})();

function polyEvalHi(poly, x) {
  let y = poly[0];
  for (let i = 1; i < poly.length; i++) y = gMul(y, x) ^ poly[i];
  return y;
}

// Encode K data bytes into an N-byte systematic codeword.
export function rsEncodeBytes(data) {
  if (data.length !== K) throw new Error(`rsEncode: expected ${K} bytes, got ${data.length}`);
  const work = new Uint8Array(N);
  work.set(data, 0);
  for (let i = 0; i < K; i++) {
    const lead = work[i];
    if (lead === 0) continue;
    const leadLog = LOG[lead];
    for (let j = 1; j < GEN.length; j++) {
      const gj = GEN[j];
      if (gj === 0) continue;
      work[i + j] ^= EXP[leadLog + LOG[gj]];
    }
  }
  const code = new Uint8Array(N);
  code.set(data, 0);
  code.set(work.subarray(K), K);
  return code;
}

// Decode an N-byte received codeword:
//   { data: Uint8Array(K), errors: number, ok: boolean }
// errors = byte positions corrected (0..t), or -1 if uncorrectable.
export function rsDecodeBytes(received) {
  if (received.length !== N) throw new Error(`rsDecode: expected ${N} bytes, got ${received.length}`);
  const code = new Uint8Array(received);

  // Syndromes
  const syn = new Uint8Array(NSYM);
  let any = false;
  for (let i = 0; i < NSYM; i++) {
    syn[i] = polyEvalHi(code, EXP[i]);
    if (syn[i] !== 0) any = true;
  }
  if (!any) return { data: code.slice(0, K), errors: 0, ok: true };

  // Berlekamp-Massey
  let sigma = [1], prev = [1];
  let L = 0, m = 1, b = 1;
  for (let r = 0; r < NSYM; r++) {
    let delta = syn[r];
    for (let i = 1; i <= L && i < sigma.length; i++) {
      if (r - i < 0) break;
      delta ^= gMul(sigma[i], syn[r - i]);
    }
    if (delta === 0) { m++; continue; }
    const T = sigma.slice();
    const coef = gMul(delta, gInv(b));
    const shifted = new Array(prev.length + m).fill(0);
    for (let i = 0; i < prev.length; i++) shifted[i + m] = gMul(prev[i], coef);
    const newLen = Math.max(sigma.length, shifted.length);
    const next = new Array(newLen).fill(0);
    for (let i = 0; i < sigma.length; i++) next[i] ^= sigma[i];
    for (let i = 0; i < shifted.length; i++) next[i] ^= shifted[i];
    sigma = next;
    if (2 * L <= r) { L = r + 1 - L; prev = T; b = delta; m = 1; }
    else { m++; }
  }
  while (sigma.length > 1 && sigma[sigma.length - 1] === 0) sigma.pop();

  const numErrors = sigma.length - 1;
  if (numErrors === 0 || numErrors > NSYM / 2) {
    return { data: code.slice(0, K), errors: -1, ok: false };
  }

  // Chien search
  const errPositions = [];
  for (let j = 0; j < N; j++) {
    const xInv = EXP[(255 - j) % 255];
    let y = 0, pow = 1;
    for (let k = 0; k < sigma.length; k++) {
      y ^= gMul(sigma[k], pow);
      pow = gMul(pow, xInv);
    }
    if (y === 0) errPositions.push(N - 1 - j);
  }
  if (errPositions.length !== numErrors) {
    return { data: code.slice(0, K), errors: -1, ok: false };
  }

  // Error evaluator ω(x) = (S(x) * σ(x)) mod x^NSYM
  const omega = new Array(NSYM).fill(0);
  for (let n = 0; n < NSYM; n++) {
    let s = 0;
    for (let i = 0; i <= n; i++) {
      const k = n - i;
      if (k < sigma.length && i < NSYM) s ^= gMul(syn[i], sigma[k]);
    }
    omega[n] = s;
  }

  // Forney for narrow-sense (first root = 0): e_j = α^j * ω(α^{-j}) / σ'(α^{-j})
  for (const pos of errPositions) {
    const j = N - 1 - pos;
    const xInv = EXP[(255 - j) % 255];
    let omegaVal = 0, p = 1;
    for (let k = 0; k < NSYM; k++) {
      omegaVal ^= gMul(omega[k], p);
      p = gMul(p, xInv);
    }
    let sigmaDeriv = 0, pd = 1;
    for (let k = 1; k < sigma.length; k++) {
      if (k & 1) sigmaDeriv ^= gMul(sigma[k], pd);
      pd = gMul(pd, xInv);
    }
    if (sigmaDeriv === 0) {
      return { data: code.slice(0, K), errors: -1, ok: false };
    }
    const xj = EXP[j % 255];
    code[pos] ^= gMul(gMul(omegaVal, xj), gInv(sigmaDeriv));
  }

  // Verify
  for (let i = 0; i < NSYM; i++) {
    if (polyEvalHi(code, EXP[i]) !== 0) {
      return { data: code.slice(0, K), errors: -1, ok: false };
    }
  }
  return { data: code.slice(0, K), errors: errPositions.length, ok: true };
}

// ===================== bit-level wrappers =====================
export function bitsToBytesMsb(bits) {
  const out = new Uint8Array(bits.length >> 3);
  for (let i = 0; i < out.length; i++) {
    let b = 0;
    for (let j = 0; j < 8; j++) b = (b << 1) | bits[(i << 3) + j];
    out[i] = b;
  }
  return out;
}

export function bytesToBitsMsb(bytes) {
  const out = new Uint8Array(bytes.length << 3);
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i];
    for (let j = 0; j < 8; j++) out[(i << 3) + j] = (b >> (7 - j)) & 1;
  }
  return out;
}

// 817 wire bits -> 1024 codeword bits.
export function eccEncode(payloadBits) {
  if (payloadBits.length !== PAYLOAD_BITS) {
    throw new Error(`eccEncode: expected ${PAYLOAD_BITS} bits, got ${payloadBits.length}`);
  }
  const padded = new Uint8Array(K * 8);
  padded.set(payloadBits, 0);
  return bytesToBitsMsb(rsEncodeBytes(bitsToBytesMsb(padded)));
}

// 1024 (possibly corrupted) codeword bits -> 817 wire bits.
//   bits / errors / ok    — same shape as rsDecodeBytes
export function eccDecode(codewordBits) {
  if (codewordBits.length !== CODEWORD_BITS) {
    throw new Error(`eccDecode: expected ${CODEWORD_BITS} bits, got ${codewordBits.length}`);
  }
  const r = rsDecodeBytes(bitsToBytesMsb(codewordBits));
  const allBits = bytesToBitsMsb(r.data);
  return { bits: allBits.slice(0, PAYLOAD_BITS), errors: r.errors, ok: r.ok };
}
