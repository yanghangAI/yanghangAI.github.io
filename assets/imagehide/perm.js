/**
 * Per-input-size permutation generator, byte-exact to PyTorch CPU.
 *
 * The INN model's `PatchBitAdapter` builds a (INN_CHANNELS, NUM_SLOTS, p)
 * balanced permutation that depends on the DWT spatial dims (h_dwt, w_dwt).
 * It's used by both encoder.scatter and decoder.gather to place watermark
 * bits at specific positions in each DWT subband.
 *
 * PyTorch recomputes this per forward call, so eval works at any input size.
 * ONNX traces bake the perm as a constant for ONE specific size, so a
 * 256-traced ONNX run at 1024×1024 leaves ~94% of the subband unmodulated.
 *
 * Fix: re-export ONNX with the perm tensor as an INPUT, and reproduce the
 * exact PyTorch perm in JS so the JS demo can feed the right tensor at any
 * size. This file is the JS reproduction.
 *
 * Verified byte-exact against PyTorch for:
 *   - 128×128 canonical (seed=0xD1FF5E^c) at ch 0
 *   - 512×512 non-canonical (seed=(0xD1FF5E^c)^(h*1000003+w)) at ch 5
 *
 * See also: inn_model.py:_make_balanced_permutation, _perm_for.
 */

// ---------- PyTorch CPU MT19937 ----------

const N = 624, M = 397;
const UMASK = 0x80000000 >>> 0;
const LMASK = 0x7fffffff >>> 0;
const MATRIX_A = 0x9908b0df >>> 0;
const FLOAT24_DIVISOR = 1 / (1 << 24);
const MASK24 = 0xFFFFFF;

function makeMT(seed) {
  const state = new Uint32Array(N);
  state[0] = seed >>> 0;
  for (let j = 1; j < N; j++) {
    const s = state[j - 1];
    state[j] = (Math.imul(1812433253, s ^ (s >>> 30)) + j) >>> 0;
  }
  return { state, next: 0, left: 1 };
}

function nextState(mt) {
  mt.left = N; mt.next = 0;
  const s = mt.state;
  let p = 0;
  for (let j = N - M; j > 0; j--, p++) {
    const u = s[p], v = s[p + 1];
    const mb = (u & UMASK) | (v & LMASK);
    s[p] = (s[p + M] ^ (((mb >>> 1) ^ ((v & 1) ? MATRIX_A : 0)) >>> 0)) >>> 0;
  }
  for (let j = M; --j > 0; p++) {
    const u = s[p], v = s[p + 1];
    const mb = (u & UMASK) | (v & LMASK);
    s[p] = (s[p + M - N] ^ (((mb >>> 1) ^ ((v & 1) ? MATRIX_A : 0)) >>> 0)) >>> 0;
  }
  const u = s[p], v = s[0];
  const mb = (u & UMASK) | (v & LMASK);
  s[p] = (s[p + M - N] ^ (((mb >>> 1) ^ ((v & 1) ? MATRIX_A : 0)) >>> 0)) >>> 0;
}

function nextU32(mt) {
  if (--mt.left === 0) nextState(mt);
  let y = mt.state[mt.next++];
  y = (y ^ (y >>> 11)) >>> 0;
  y = (y ^ ((y << 7) & 0x9d2c5680)) >>> 0;
  y = (y ^ ((y << 15) & 0xefc60000)) >>> 0;
  y = (y ^ (y >>> 18)) >>> 0;
  return y;
}

// PyTorch at::uniform_real<float>: x = (val & ((1<<24)-1)) / (1<<24)
function nextFloat32(mt) {
  return (nextU32(mt) & MASK24) * FLOAT24_DIVISOR;
}

// ---------- balanced permutation (one channel) ----------

// Returns a flat Int32Array of length numPositions: a permutation of [0, numPositions).
// Equivalent to torch.argsort(torch.rand(numPositions, generator=g)) with g seeded
// from `seed & 0xFFFFFFFF` (PyTorch masks to low 32 bits internally).
function balancedPermOneChannel(numPositions, seed) {
  const mt = makeMT(seed >>> 0);
  // Float64 to avoid losing precision when many duplicate float32 values land
  // in nearby buckets — JS Array.sort is stable so ties resolve by insertion
  // index (which matches PyTorch's stable sort default).
  const vals = new Float64Array(numPositions);
  for (let i = 0; i < numPositions; i++) vals[i] = nextFloat32(mt);
  // For large N, building a typed-index array + JS sort is the cheapest path
  // (TypedArray sort is heavily optimized in V8/Spidermonkey).
  const idx = new Int32Array(numPositions);
  for (let i = 0; i < numPositions; i++) idx[i] = i;
  // Sort idx by vals[idx[i]]. V8 sorts TypedArrays with a numeric comparator
  // in O(N log N) using TimSort, stable.
  const sorted = Array.from(idx).sort((a, b) => vals[a] - vals[b]);
  for (let i = 0; i < numPositions; i++) idx[i] = sorted[i];
  return idx;
}

// ---------- public API ----------

// Constants from inn_model.py.
export const INN_CHANNELS = 12;
export const NUM_SLOTS = 1024;          // GRID*GRID = 32*32

// Canonical regime: the model treats 256×256 input specially (h_dwt=128,
// w_dwt=128) and uses a seed that does NOT include the size mixin. Other
// sizes use the size-mixed seed.
const CANONICAL_H_DWT = 128;
const CANONICAL_W_DWT = 128;
const CANONICAL_SEED_C0 = 0xD1FF5E;

function _seedForChannel(channel, hDwt, wDwt) {
  if (hDwt === CANONICAL_H_DWT && wDwt === CANONICAL_W_DWT) {
    return (CANONICAL_SEED_C0 ^ channel) & 0xFFFFFFFF;
  }
  // (0xD1FF5E ^ c) ^ (h_dwt * 1000003 + w_dwt) — JS bitwise wraps to int32,
  // which matches PyTorch's seed & ((1<<63)-1) followed by MT's low-32 mask.
  // We compute mod 2^32 explicitly because h_dwt*1000003 can overflow int32.
  const sizeMix = ((hDwt * 1000003) >>> 0) + wDwt;  // mod 2^32 wrap via >>>0 after
  const baseXorC = (CANONICAL_SEED_C0 ^ channel) >>> 0;
  return (baseXorC ^ (sizeMix >>> 0)) >>> 0;
}

/**
 * Build the (INN_CHANNELS, NUM_SLOTS, p) perm stack for a given DWT size,
 * returned as a single flat Int32Array of length INN_CHANNELS*NUM_SLOTS*p
 * laid out [channel, slot, p-element] in row-major order — i.e. exactly the
 * memory layout ONNX expects for the input tensor of shape
 * (INN_CHANNELS, NUM_SLOTS, p).
 *
 * p = (h_dwt * w_dwt) / NUM_SLOTS — must be divisible.
 */
export function buildPermStack(hDwt, wDwt) {
  const numPositions = hDwt * wDwt;
  if (numPositions % NUM_SLOTS !== 0) {
    throw new Error(
      `h_dwt*w_dwt = ${numPositions} must be a multiple of NUM_SLOTS=${NUM_SLOTS}`);
  }
  const p = numPositions / NUM_SLOTS;
  const out = new Int32Array(INN_CHANNELS * NUM_SLOTS * p);
  for (let c = 0; c < INN_CHANNELS; c++) {
    const seed = _seedForChannel(c, hDwt, wDwt);
    const perm = balancedPermOneChannel(numPositions, seed);
    // perm is length numPositions = NUM_SLOTS * p. Already in the right
    // order for reshape(NUM_SLOTS, p), so just copy block-wise.
    out.set(perm, c * numPositions);
  }
  return { data: out, channels: INN_CHANNELS, slots: NUM_SLOTS, p };
}

/**
 * Compute the DWT-1 dims for an INN input. INN does a single-level Haar DWT
 * so DWT halves spatial dims. Inputs must be even-sized (the model trim
 * already rounds down to multiples of 64).
 */
export function dwtDims(inputH, inputW) {
  return { hDwt: inputH >> 1, wDwt: inputW >> 1 };
}
