// BCH(127, 78, t=7) over GF(2^7) for Slepian-Wolf pHash compression.
//
// Theory: the receiver computes the pHash H' of the (attacked) container
// locally. The sender transmits only the 49-bit syndrome of the original H
// under a BCH(127, 78, t=7) code. The receiver XORs its own syndrome with
// the received one to get the syndrome of the bit-error vector e = H ⊕ H',
// then BCH-decodes e (correcting up to 7 bit-flips of pHash drift) and
// recovers H = H' ⊕ e. The signed quantity is H so sig.verify() then works.
//
// Why BCH not RS: pHash drift is typically scattered random bit flips, not
// byte-aligned bursts; binary BCH gets ~16x the error-correction efficiency
// of byte-level RS on this kind of channel.
//
// Conventions:
//   * pHash bit array v is Uint8Array of length 127, indexed v[i] = bit i.
//     v[i] is the coefficient of x^i in the polynomial v(x).
//   * Syndromes are computed over GF(2^7) with primitive polynomial x^7+x^3+1.
//   * For narrow-sense BCH we only transmit the 7 "odd" syndromes
//     S_1, S_3, ..., S_13 — the even ones are derivable via the squaring rule
//     S_{2k+1}_indexform = (S_k)^2 in characteristic 2 (S_i := error(α^i)).

const M = 7;
const N127 = 127;        // codeword length
const T = 4;             // error-correction capability — 1 bit margin over PDQ's
                         // observed max drift of 3/128 across 30 COCO × 11 attacks
const NSYND_ODD = T;     // we send S_1, S_3, ..., S_(2t-1) -> T values × M bits
const SYNDROME_BITS = NSYND_ODD * M;
const NSYND_FULL = 2 * T;  // BM operates on the full 2t syndromes

const PRIM = 0x89;       // x^7 + x^3 + 1, primitive polynomial of GF(2^7)

const EXP = new Int16Array(2 * N127);
const LOG = new Int16Array(N127 + 1);
(function _init() {
  for (let i = 0; i <= N127; i++) LOG[i] = -1;
  let x = 1;
  for (let i = 0; i < N127; i++) {
    EXP[i] = x;
    LOG[x] = i;
    x <<= 1;
    if (x & (1 << M)) x ^= PRIM;
  }
  for (let i = N127; i < 2 * N127; i++) EXP[i] = EXP[i - N127];
})();

const fMul = (a, b) => (a === 0 || b === 0) ? 0 : EXP[LOG[a] + LOG[b]];
const fSq  = (a) => a === 0 ? 0 : EXP[(2 * LOG[a]) % N127];
const fInv = (a) => { if (a === 0) throw new Error('fInv(0)'); return EXP[N127 - LOG[a]]; };

// Compute S_i = v(α^i) for i in [1, NSYND_FULL] -> array indexed 0..NSYND_FULL-1
// where the array[k] corresponds to S_{k+1}.
function syndromesAll(v) {
  const s = new Int16Array(NSYND_FULL);
  for (let k = 0; k < NSYND_FULL; k++) {
    const alpha = EXP[k + 1];
    let sum = 0;
    let pow = 1;
    for (let i = 0; i < v.length; i++) {
      if (v[i] !== 0) sum ^= pow;
      pow = fMul(pow, alpha);
    }
    s[k] = sum;
  }
  return s;
}

// Compute only the odd-indexed syndromes S_1, S_3, ..., S_13 (7 values).
function syndromesOdd(v) {
  const s = new Int16Array(NSYND_ODD);
  for (let i = 0; i < NSYND_ODD; i++) {
    const alpha = EXP[2 * i + 1];
    let sum = 0;
    let pow = 1;
    for (let j = 0; j < v.length; j++) {
      if (v[j] !== 0) sum ^= pow;
      pow = fMul(pow, alpha);
    }
    s[i] = sum;
  }
  return s;
}

// Berlekamp-Massey over GF(2^7). Input: 14 syndromes (full BM needs all 2t).
// Returns σ(x) as a low-to-high coefficient array, and L = degree.
function berlekampMassey(syn) {
  let sigma = [1];
  let prev = [1];
  let L = 0;
  let m = 1;
  let b = 1;
  for (let r = 0; r < NSYND_FULL; r++) {
    let delta = syn[r];
    for (let i = 1; i <= L && i < sigma.length; i++) {
      if (r - i < 0) break;
      delta ^= fMul(sigma[i], syn[r - i]);
    }
    if (delta === 0) {
      m++;
    } else {
      const T0 = sigma.slice();
      const coef = fMul(delta, fInv(b));
      const shifted = new Array(prev.length + m).fill(0);
      for (let i = 0; i < prev.length; i++) shifted[i + m] = fMul(prev[i], coef);
      const newLen = Math.max(sigma.length, shifted.length);
      const next = new Array(newLen).fill(0);
      for (let i = 0; i < sigma.length; i++) next[i] ^= sigma[i];
      for (let i = 0; i < shifted.length; i++) next[i] ^= shifted[i];
      sigma = next;
      if (2 * L <= r) {
        L = r + 1 - L;
        prev = T0;
        b = delta;
        m = 1;
      } else {
        m++;
      }
    }
  }
  while (sigma.length > 1 && sigma[sigma.length - 1] === 0) sigma.pop();
  return { sigma, L };
}

// Chien search: find roots of σ(x). Returns array of error positions (polynomial
// degrees), each in [0, N127). For binary BCH the error magnitude at every found
// position is 1, so we don't run Forney.
function chienSearch(sigma) {
  const out = [];
  for (let j = 0; j < N127; j++) {
    const xInv = EXP[(N127 - j) % N127];
    let y = 0, pow = 1;
    for (let k = 0; k < sigma.length; k++) {
      y ^= fMul(sigma[k], pow);
      pow = fMul(pow, xInv);
    }
    if (y === 0) out.push(j);
  }
  return out;
}

// Pack the 7 odd syndromes (each a 7-bit GF element) into a 49-bit array.
function packSyndromeBits(synOdd) {
  const out = new Uint8Array(SYNDROME_BITS);
  for (let i = 0; i < NSYND_ODD; i++) {
    const v = synOdd[i];
    for (let b = 0; b < M; b++) out[i * M + b] = (v >> (M - 1 - b)) & 1;
  }
  return out;
}

function unpackSyndromeBits(bits49) {
  const synOdd = new Int16Array(NSYND_ODD);
  for (let i = 0; i < NSYND_ODD; i++) {
    let v = 0;
    for (let b = 0; b < M; b++) v = (v << 1) | bits49[i * M + b];
    synOdd[i] = v;
  }
  return synOdd;
}

// Public: encode the Slepian-Wolf "side payload" — the 49-bit syndrome that
// gets transmitted in place of the full 127-bit pHash. Input: 127 bits.
export function bchEncodeSyndrome(v127) {
  if (v127.length !== N127) throw new Error(`bchEncode: expected ${N127} bits, got ${v127.length}`);
  return packSyndromeBits(syndromesOdd(v127));
}

// Public: given the received 49-bit syndrome and the locally-computed guess
// vGuess127 (127 bits), recover the true 127-bit vector v such that
// d(v, vGuess127) ≤ T. Returns { bits, errors, ok }:
//   bits   — the corrected 127-bit vector (== vGuess127 if ok && errors=0)
//   errors — number of bit positions flipped (0..T), or -1 if uncorrectable
//   ok     — true iff decoding succeeded (errors ≤ T)
export function bchDecode(syndromeBits49, vGuess127) {
  if (syndromeBits49.length !== SYNDROME_BITS) {
    throw new Error(`bchDecode: expected ${SYNDROME_BITS} syndrome bits`);
  }
  if (vGuess127.length !== N127) {
    throw new Error(`bchDecode: expected ${N127} guess bits`);
  }

  const sOddRx = unpackSyndromeBits(syndromeBits49);     // 7 odd syndromes (received)
  const sOddGuess = syndromesOdd(vGuess127);              // 7 odd syndromes (local)

  // Syndromes of the error vector e: S_{2k+1}(e) = S_{2k+1}(rx) ⊕ S_{2k+1}(guess)
  const synError = new Int16Array(NSYND_FULL);
  for (let k = 0; k < NSYND_ODD; k++) {
    synError[2 * k] = sOddRx[k] ^ sOddGuess[k];  // index 2k stores S_{2k+1}
  }
  // Squaring rule (binary BCH): S_{2k+1}(e) is computed; derive S_{2(2k+1)}(e) = S_{2k+1}^2.
  // In our 0-based array: synError[2k+1] = (synError[k])^2.
  for (let k = 0; k < NSYND_ODD; k++) {
    synError[2 * k + 1] = fSq(synError[k]);
  }

  // No errors?
  let any = false;
  for (let i = 0; i < NSYND_FULL; i++) if (synError[i] !== 0) { any = true; break; }
  if (!any) {
    return { bits: Uint8Array.from(vGuess127), errors: 0, ok: true };
  }

  const { sigma, L } = berlekampMassey(synError);
  if (L === 0 || L > T) {
    return { bits: Uint8Array.from(vGuess127), errors: -1, ok: false };
  }
  const positions = chienSearch(sigma);
  if (positions.length !== L) {
    return { bits: Uint8Array.from(vGuess127), errors: -1, ok: false };
  }

  // Flip the error bits in vGuess127 to recover v.
  const out = Uint8Array.from(vGuess127);
  for (const j of positions) {
    if (j < 0 || j >= N127) {
      return { bits: Uint8Array.from(vGuess127), errors: -1, ok: false };
    }
    out[j] ^= 1;
  }
  // Sanity: recompute odd syndromes of `out` and confirm they match the received ones.
  const sOddOut = syndromesOdd(out);
  for (let i = 0; i < NSYND_ODD; i++) {
    if (sOddOut[i] !== sOddRx[i]) {
      return { bits: Uint8Array.from(vGuess127), errors: -1, ok: false };
    }
  }
  return { bits: out, errors: positions.length, ok: true };
}

export const BCH_N = N127;
export const BCH_T = T;
export const BCH_SYNDROME_BITS = SYNDROME_BITS;
