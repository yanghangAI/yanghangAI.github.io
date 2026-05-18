# imagehide demo — handoff

What the demo does, how it's wired, what we got wrong, what to watch out for.

## What it is

Browser demo at `/imagehide-demo/`: encode an invisible 896-bit payload
(or 96-byte text message) into an image with a small INN watermarking
model, then decode it back after a channel attack (JPEG, resize, social-
media chains). Everything runs client-side in a Web Worker.

- **Model**: 1024-bit `INNCodec` from
  `/work/pi_nwycoff_umass_edu/hang/autohide/experiments/1024bit_noamp_keepckpt/idea097_design001/`
  (PyTorch checkpoint `ckpt_latest.pt`). FP16 ONNX exports in
  `assets/imagehide/{encoder,decoder}.onnx`. ~0.6 / 1.1 MB each.
- **Wire payload (796 bits)**: BCH(127, t=4) syndrome of pHash (28b) +
  Ed25519 sig (512b) + pk (256b). RS(128, 100) over GF(2^8) wraps it
  into the 1024-bit codeword the model embeds. See
  [`ECC_DESIGN.md`](ECC_DESIGN.md) for the Slepian-Wolf-trick rationale.
- **pHash**: full PDQ (`pdq.js`) — empirical max drift 3/128 across 30
  COCO × 11 attacks.

## File layout

| file | purpose |
|---|---|
| `app.js` | UI, encode/decode flow, ECC + BCH stages, perm cache |
| `pipeline.js` | main-thread ↔ worker bridge, model fetch, mode switch |
| `decoder-worker.js` | ORT-Web session host. Encoder + decoder, single session at a time |
| `perm.js` | JS port of PyTorch CPU MT19937 + `_make_balanced_permutation` (see "BIG MISTAKE" below) |
| `attacks.js` | resize / JPEG / social-platform chain attacks on Float32 frames |
| `ecc.js` | RS(128, 100) over GF(2^8) |
| `bch.js` | BCH(127, t=4) Slepian-Wolf syndrome for pHash compression |
| `pdq.js` | PDQ perceptual hash (Jarosz + DCT + zigzag) |
| `payload.js` | bit/byte packing, signature glue |
| `trim.js` | crop input to a multiple of 64 (model needs it) |
| `metrics.js` | PSNR, SSIM for the encode-result display |

## The BIG MISTAKE that crushed bit accuracy

**Symptom**: demo's `chain_wechat` bit-acc was ~0.80 while the eval at
1024×1024 showed ~0.99. Clean bit-acc was near 1.0 in both. PSNR was
also suspiciously *higher* in the demo than eval (~40 dB vs ~37 dB).

**Cause**: The model's `PatchBitAdapter` builds a per-input-size
permutation in `_perm_for(h_dwt, w_dwt)` via a seeded PRNG —
`torch.argsort(torch.rand(h_dwt*w_dwt, generator=g))`. The seed is
`(0xD1FF5E ^ channel) ^ (h_dwt*1000003 + w_dwt)`, so it's different at
every input size. PyTorch eval recomputes this perm every forward call,
covering the full DWT subband. **`torch.onnx.export` traces one
concrete forward and freezes the perm tensor that ran during the trace
as literal constants in `ScatterElements`.**

The exports were at H=W=256 (`PROBE_HW=256`), so the baked permutation
indexed positions `[0, 16384)` — the right perm for a 128×128 DWT
subband at that input size. At demo input 1024×1024 the DWT subband is
512×512 = 262144 positions; the baked indices still address `[0..16384)`,
which is **only the top 32 rows (~6 %) of the subband**.

Consequences:
- Clean bit-acc stays near 1.0 because the bits are reliably embedded
  at *those specific* 16384 positions.
- Aggressive attacks (resize 0.3 → low-Q JPEG) destroy that small strip
  disproportionately, killing bit-acc.
- PSNR is artificially boosted because the residual is concentrated in
  6 % of the area — average distortion across the whole image is lower.

**Fix** (commits `d212697`, `029076a`):

1. **Re-export ONNX with the perm as an input tensor.** New script:
   `experiments/.../export_onnx_perm_input.py`. The encoder takes
   `(host_rgb, bits, perm_stack)` and the decoder takes
   `(container_rgb, perm_stack)`. Inside the export wrapper we
   monkey-patch `codec.adapter._perm_for` to return the input tensor;
   the traced graph then references that input wherever it used to bake
   constants.

2. **Port PyTorch's perm to JS exactly.** `assets/imagehide/perm.js`
   reproduces PyTorch CPU MT19937 + `at::uniform_real<float>` + argsort
   byte-for-byte. Verified against PyTorch reference values on
   (128×128, ch 0) and (512×512, ch 5). The trick was that PyTorch's
   `at::uniform_real<float>` uses the **bottom** 24 bits of the uint32
   (`val & 0xFFFFFF`), not the top 24 — I got that wrong on the first
   try and got plausible-but-wrong floats.

3. **Compute the perm in JS per input size**, cache by `(h_dwt, w_dwt)`,
   pass into encoder/decoder via the worker as an int64 tensor.

After the fix: demo `chain_wechat` matches eval at ~0.99. PSNR also
drops to the eval-realistic ~37 dB (the residual now spreads across
the full subband, so per-pixel distortion is uniform, not concentrated).

### Lesson

> If a PyTorch model has any computation that depends on input shape via
> Python (RNG-seeded perms, dynamically-built index tensors, Python-side
> caches), `torch.onnx.export` will bake the trace-time concrete value
> as a constant. CNN convs / DWT / pooling all generalize because
> they're truly polymorphic ops, but anything routed through Python
> code at forward time gets frozen. Either pass the dynamic value as an
> input tensor, or trace at every input size you'll ever run.

## Secondary mistakes worth knowing

These didn't tank bit-acc but each was a quiet ~few % loss:

1. **Canvas resize used `'medium'` smoothing** (Mitchell-Netravali
   anti-aliasing). PyTorch eval uses pure bilinear without anti-aliasing
   (PyTorch's `F.interpolate(mode='bilinear', align_corners=False)`
   aliases on downsample by design). The Mitchell low-pass destroyed
   the high-frequency band the watermark encodes in. **Fixed** by
   implementing a pure-JS bilinear that matches PyTorch exactly
   (`attacks.js` → `resizeOneAxisF32`), verified against reference
   values on a 4×4 → 2×2 → 4×4 round-trip.

2. **Encoder float32 output was quantized to uint8 too early**, before
   the canvas resize. The encoder produces residuals at sub-uint8
   magnitude (~±5/255 typically, many pixels < 1/255); quantizing
   before the resize destroyed sub-quantum information that the
   bilinear filter would otherwise have preserved by averaging
   neighbors. **Fixed** by keeping the container Float32 from encoder
   through the JS resize, only quantizing to uint8 at the JPEG canvas
   boundary (where it has to be uint8 anyway).

3. **WebGPU op-coverage caveat**: ORT-Web 1.20 WebGPU has limited op
   coverage; some shape-related ops always run on the WASM execution
   provider as fallback. Harmless but generates the "Some nodes were
   not assigned to the preferred execution providers" warning at
   session create. Ignore it.

## Memory & backend story (the long one)

The other ~30 commits in the recent history were spent chasing iPhone
Safari memory issues. Net state:

- **macOS / desktop**: WebGPU (FP16 model, ~5-20× faster than WASM).
- **iPhone Safari**: WASM only. We load `ort.min.mjs` (no WebGPU code
  at all) and pass `executionProviders: ['wasm']`. Reasons documented
  in `d41d474`'s commit message.
- **Android**: still tries WebGPU; falls through to error if it fails
  (no WASM fallback configured).

### What we tried on iPhone and gave up on

In order of attempt:
- Mixed `['webgpu', 'wasm']` providers → WebGPU partial failure
  poisoned WASM init (`d41d474` commit message has the full chain).
- Worker terminate-respawn between modes → iOS pins GPU memory tied to
  dead workers; budget ate up across runs.
- `session.release()` per N runs (periodic refresh, codex-suggested) →
  helped some but ORT-internal GPU buffers still accumulated.
- `GPUDevice.destroy()` between sessions → freed memory but broke
  ORT's next `session.create()` with "Range consisting of offset and
  length are out of bounds".
- Disabling `enableMemPattern` / `enableCpuMemArena` → backwards,
  forces more allocations per run, not fewer.
- Various sleep durations between teardown and re-create.

iPhone Safari's per-tab WebGPU pool is ~100 MB. One inference of our
INN at 0.5 MP is ~50-100 MB. ORT-Web 1.20's WebGPU backend doesn't
expose a way to truly release GPU memory between runs from JS, so
runs accumulate and the tab dies after 3-6 cycles. **Conclusion**:
WebGPU on iPhone Safari isn't currently viable for this model. WASM
at 0.5 MP works through many decodes.

### Current memory caps

- Desktop: 1 MP input cap. WebGPU.
- Mobile (incl. iPhone): 0.5 MP. WASM.

### Worker lifecycle

ONE worker for the page lifetime. Mode switches between encoder and
decoder happen *in place*: the worker calls `session.release()` then
creates a new session — no `worker.terminate()` between modes. This
was critical for iPhone where terminate leaves GPU/WASM memory pinned.
Periodic session refresh runs every 4 inferences on iOS, 6 on Android,
30 on desktop (`REFRESH_EVERY` in `decoder-worker.js`).

## How to redeploy

GitHub Pages auto-deploys on push to `master`. Jekyll fills
`site.time` into the cache-bust query (`?v=${unix_timestamp}`), so each
deploy invalidates the previous JS bundles. The build timestamp is
shown in the status bar at the right (`build YYYY-MM-DD HH:MM UTC ·
vNNNNNN`) so you can confirm the deploy went live by reloading and
checking the badge.

⚠ The cron job `scripts/update-cluster-dashboard.sh` runs
`git fetch && git checkout && git reset --hard origin/master` every 15
minutes on the working directory. Any local edits that aren't committed
and pushed within that window get wiped. **Commit + push immediately
after every edit.**

## How to re-export ONNX (if you change the model)

```bash
cd /work/pi_nwycoff_umass_edu/hang/autohide/experiments/1024bit_noamp_keepckpt/idea097_design001/
sbatch export_onnx_perm_input.sbatch       # FP32 perm-input, ~5 min
sbatch export_onnx_perm_input_fp16.sbatch  # FP16 wholesale enc + selective dec, ~10-20 min
# Copy the FP16 versions to the demo:
cp onnx_out/{encoder,decoder}_perm_input_fp16.onnx /home/hangyang_umass_edu/yanghangAI.github.io/assets/imagehide/{encoder,decoder}.onnx
git -C /home/hangyang_umass_edu/yanghangAI.github.io add assets/imagehide/{encoder,decoder}.onnx
git -C /home/hangyang_umass_edu/yanghangAI.github.io commit -m "..." && git -C /home/hangyang_umass_edu/yanghangAI.github.io push
```

Validation thresholds in the FP16 export: encoder max-abs error < 0.05,
decoder sign-flips == 0 on a multi-size probe bank (64 / 128 / 256).
The selective per-init harm scan takes the bulk of the 10-20 min budget.

## Open questions / future work

- **iPhone WebGPU**: probably needs ORT-Web 1.21+ (some WebGPU memory
  management fixes shipped) or a smaller model variant to ever work.
  Not blocking — WASM works.
- **Smaller model**: INT8 quantization or pruning could roughly halve
  WASM memory pressure and maybe make iPhone WebGPU viable. Training-
  side work, not deployment-side.
- **`chain_wechat_full` (resize + JPEG + Gaussian noise σ=0.03)** in
  eval drops bit-acc to ~0.77 even at 1024×1024. The demo doesn't
  currently include this attack, but if the user uploads a noisy image
  from the wild we'd see similar degradation. Mention it if asked.
- **libsodium 404**: `https://cdn.jsdelivr.net/npm/libsodium-wrappers@0.7.13/+esm`
  occasionally 404s on iPhone Safari (the `+esm` endpoint is jsDelivr
  magic). Auto-mode signature verification fails when this happens.
  Custom-text mode (no signature) works regardless. Worth pinning a
  specific working URL if it becomes a problem.

## Top-of-mind invariants

1. Don't terminate the worker between encode and decode (iPhone pins
   the dead worker's memory).
2. Don't disable `enableMemPattern` (ORT's buffer reuse helps, not
   hurts).
3. Don't load `ort.webgpu.min.mjs` on iPhone (touches GPU at import).
4. The perm passed to the model MUST match what PyTorch would compute
   for that `(h_dwt, w_dwt)` — bit-for-bit. Any deviation produces
   plausible-looking but worse-than-eval bit-acc.
5. Encoder Float32 output stays Float32 through the JS resize. Quantize
   only at the JPEG canvas boundary.
