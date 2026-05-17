// Reed-Solomon RS(128, 112) over GF(2^8) with primitive polynomial 0x11D
// (the standard x^8 + x^4 + x^3 + x^2 + 1 used in Reed-Solomon for QR codes,
// CCSDS, etc).
//
//   Data:    112 bytes (= 896 user bits)
//   Parity:  16 bytes  (= 128 redundancy bits)
//   Total:   128 bytes (= 1024 codeword bits — matches the new INN model's
//                       1024-bit input/output channel)
//
// Corrects up to floor(16/2) = 8 byte errors anywhere in the codeword.
// Detects more errors than that (uncorrectable) without falsely "correcting"
// to a wrong codeword in most cases.
//
// Conventions used throughout:
//   * Codeword byte layout: code[0..K-1] = data bytes, code[K..N-1] = parity
//   * Polynomial interpretation: code[i] is the coefficient of x^(N-1-i),
//     so code[0] is the highest-degree coefficient
//   * Generator polynomial: narrow-sense, g(x) = ∏(x - α^i) for i in [0, NSYM)
//   * Berlekamp-Massey σ stored low-to-high (sigma[0] = constant term = 1)

export const N = 128;
export const K = 112;
export const NSYM = N - K;          // 16 parity bytes; t = NSYM/2 = 8 corrections

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
  // Mirror the EXP table so we never need a modulo in gfMul.
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
      next[j] ^= g[j];                // x * coefficient
      next[j + 1] ^= gMul(g[j], ai);  // α^i * coefficient
    }
    g = next;
  }
  return g;
})();

// Horner evaluation. poly[0] is the highest-degree coefficient.
function polyEvalHi(poly, x) {
  let y = poly[0];
  for (let i = 1; i < poly.length; i++) y = gMul(y, x) ^ poly[i];
  return y;
}

// Encode K data bytes into an N-byte systematic codeword:
//   code[0..K-1] = data (unchanged)
//   code[K..N-1] = parity bytes
export function rsEncodeBytes(data) {
  if (data.length !== K) throw new Error(`rsEncode: expected ${K} bytes, got ${data.length}`);
  // Long-division of (data shifted left by NSYM positions, i.e. data followed by
  // NSYM zeros) by GEN. The remainder ends up in positions [K, N).
  const work = new Uint8Array(N);
  work.set(data, 0);
  for (let i = 0; i < K; i++) {
    const lead = work[i];
    if (lead === 0) continue;
    const leadLog = LOG[lead];
    // GEN[0] = 1 in our construction, so dividing by it is a no-op.
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

// Decode an N-byte received codeword. Returns:
//   { data: Uint8Array(K), errors: number, ok: boolean }
// errors = number of byte positions corrected (0..t), or -1 if uncorrectable.
export function rsDecodeBytes(received) {
  if (received.length !== N) throw new Error(`rsDecode: expected ${N} bytes, got ${received.length}`);
  const code = new Uint8Array(received);   // working copy

  // 1. Syndromes: S_i = code(α^i) for i in [0, NSYM)
  const syn = new Uint8Array(NSYM);
  let anyNonzero = false;
  for (let i = 0; i < NSYM; i++) {
    syn[i] = polyEvalHi(code, EXP[i]);
    if (syn[i] !== 0) anyNonzero = true;
  }
  if (!anyNonzero) {
    return { data: code.slice(0, K), errors: 0, ok: true };
  }

  // 2. Berlekamp-Massey to find σ(x). Low-to-high coefficient layout here.
  //    σ has σ[0] = 1; degree of σ = number of errors.
  let sigma = [1];
  let prev = [1];
  let L = 0;
  let m = 1;
  let b = 1;
  for (let r = 0; r < NSYM; r++) {
    let delta = syn[r];
    for (let i = 1; i <= L && i < sigma.length; i++) {
      if (r - i < 0) break;
      delta ^= gMul(sigma[i], syn[r - i]);
    }
    if (delta === 0) {
      m++;
    } else {
      const T = sigma.slice();
      const coef = gMul(delta, gInv(b));
      const shifted = new Array(prev.length + m).fill(0);
      for (let i = 0; i < prev.length; i++) shifted[i + m] = gMul(prev[i], coef);
      const newLen = Math.max(sigma.length, shifted.length);
      const next = new Array(newLen).fill(0);
      for (let i = 0; i < sigma.length; i++) next[i] ^= sigma[i];
      for (let i = 0; i < shifted.length; i++) next[i] ^= shifted[i];
      sigma = next;
      if (2 * L <= r) {
        L = r + 1 - L;
        prev = T;
        b = delta;
        m = 1;
      } else {
        m++;
      }
    }
  }
  // Trim trailing zeros so sigma.length - 1 is the true degree.
  while (sigma.length > 1 && sigma[sigma.length - 1] === 0) sigma.pop();

  const numErrors = sigma.length - 1;
  if (numErrors === 0 || numErrors > NSYM / 2) {
    return { data: code.slice(0, K), errors: -1, ok: false };
  }

  // 3. Chien search: find positions j (polynomial degree) where σ(α^{-j}) = 0.
  //    Byte index in `code` is pos = N-1-j.
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

  // 4. Error evaluator ω(x) = (S(x) * σ(x)) mod x^NSYM,  low-to-high.
  //    S(x) = syn[0] + syn[1] x + ... + syn[NSYM-1] x^(NSYM-1).
  const omega = new Array(NSYM).fill(0);
  for (let n = 0; n < NSYM; n++) {
    let s = 0;
    for (let i = 0; i <= n; i++) {
      const k = n - i;
      if (k < sigma.length && i < NSYM) s ^= gMul(syn[i], sigma[k]);
    }
    omega[n] = s;
  }

  // 5. Forney for narrow-sense (first root = 0):
  //    e_j = α^j * ω(α^{-j}) / σ'(α^{-j})
  for (const pos of errPositions) {
    const j = N - 1 - pos;
    const xInv = EXP[(255 - j) % 255];

    let omegaVal = 0, p = 1;
    for (let k = 0; k < NSYM; k++) {
      omegaVal ^= gMul(omega[k], p);
      p = gMul(p, xInv);
    }
    // Formal derivative in characteristic 2: σ'(x) = Σ σ[k] x^(k-1) for odd k.
    let sigmaDeriv = 0, pd = 1;
    for (let k = 1; k < sigma.length; k++) {
      if (k & 1) sigmaDeriv ^= gMul(sigma[k], pd);
      pd = gMul(pd, xInv);
    }
    if (sigmaDeriv === 0) {
      return { data: code.slice(0, K), errors: -1, ok: false };
    }
    const xj = EXP[j % 255];
    const eMag = gMul(gMul(omegaVal, xj), gInv(sigmaDeriv));
    code[pos] ^= eMag;
  }

  // 6. Verify: recompute syndromes on the corrected codeword. Any non-zero means
  //    BM/Chien locked onto a phantom error pattern — declare uncorrectable.
  for (let i = 0; i < NSYM; i++) {
    if (polyEvalHi(code, EXP[i]) !== 0) {
      return { data: code.slice(0, K), errors: -1, ok: false };
    }
  }

  return { data: code.slice(0, K), errors: errPositions.length, ok: true };
}

// ============================ bit-level wrappers ============================
// The demo carries bits as Uint8Array(0/1) in MSB-first byte order. Convert at
// the boundary so the RS code can operate on bytes.

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

// 896 user bits -> 1024 codeword bits
export function eccEncode(bits896) {
  if (bits896.length !== 896) throw new Error(`eccEncode: expected 896 bits, got ${bits896.length}`);
  return bytesToBitsMsb(rsEncodeBytes(bitsToBytesMsb(bits896)));
}

// 1024 (possibly corrupted) codeword bits -> 896 user bits
//   bits:      the recovered 896-bit user payload
//   errors:    number of byte positions corrected, or -1 if uncorrectable
//   ok:        false iff errors > t (decoder gave up)
export function eccDecode(bits1024) {
  if (bits1024.length !== 1024) throw new Error(`eccDecode: expected 1024 bits, got ${bits1024.length}`);
  const r = rsDecodeBytes(bitsToBytesMsb(bits1024));
  return { bits: bytesToBitsMsb(r.data), errors: r.errors, ok: r.ok };
}
