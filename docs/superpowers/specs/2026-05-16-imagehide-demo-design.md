# Imagehide Demo Page — Design

**Date:** 2026-05-16
**Site repo:** `yanghangAI/yanghangAI.github.io`
**Model repo:** `/work/pi_nwycoff_umass_edu/hang/imagehide` (local on Unity HPC)
**Goal:** A user-upload interactive demo of the MWIP (Minimal Whole-Image Provenance) INN watermark at `https://yanghangAI.github.io/imagehide/`. User uploads any photo → 896-bit payload embedded → photo runs through 11 channel attacks → bit accuracy, signature verify, PSNR/SSIM displayed per attack. All client-side.

## Audience & purpose

Two audiences:
1. **Reviewers and academic readers** of the MWIP work who want to see the watermark work, not just read about it.
2. **The author** as a portfolio artifact: live demo of the deployed model is more compelling than static result images.

## Architecture overview

```
Browser
  ├─ /imagehide/                 page shell (academic plain HTML/CSS)
  ├─ widget HTML + scoped CSS
  ├─ ONNX Runtime Web (CDN, lazy)        WebGPU + WASM fallback
  ├─ libsodium-wrappers (CDN, lazy)      Ed25519 sign/verify
  ├─ Canvas API                          JPEG round-trip, bilinear resize
  └─ Static assets:
        assets/imagehide/encoder.onnx    INN forward, ~8 MB fp16
        assets/imagehide/decoder.onnx    INN reverse z=0, ~8 MB fp16
        assets/imagehide/sample-cover.jpg  "Load default sample" target
```

No backend. Everything ships from yanghangAI.github.io.

## Inference path

The trained model exposes two ops (`src/inn_model.py`):

- `INNCodec.embed(host_rgb[-1,1], bits) -> container_rgb[-1,1]`
- `INNCodec.extract(container_rgb[-1,1]) -> logits[896]`

Both operate on RGB tensors in `[-1, 1]` at native model resolution. Dimensions must be divisible by 64 (`_check_dims` enforces `H % (2 * GRID) == 0`, `GRID = 32`).

We export two separate ONNX graphs — `encoder.onnx` wrapping `embed`, `decoder.onnx` wrapping `extract` — both with dynamic spatial dims so any divisible-by-64 input works.

## Resolution handling

User uploads any resolution. The page:

1. Decodes to RGBA on a canvas.
2. Computes `cropH = H - (H % 64)`, `cropW = W - (W % 64)`.
3. **Center-trims** the trimmed strips (top/bottom or left/right) and stores them as untouched pixel data.
4. Encodes only the trimmed core.
5. After encoding, pastes the trimmed core back into a full-size canvas so the user receives a same-resolution output. Untouched strips are bit-identical to the input.
6. Decoder runs on the same canonical center-trim of the (possibly attacked) container.
7. PSNR/SSIM are computed over the encoded region only (the untouched strips would skew the metric trivially).

**Memory cap:** above ~6 MP (e.g. > 3072 × 2048), the page surfaces a "this will be slow / may OOM in your browser" warning with a one-click "run anyway" override.

## Payload

Fixed 896 bits: `[ H(128) | sig(512) | pk(256) ]`.

- **H (128 bits):** perceptual hash of the trimmed image, computed in JS via a simple pHash variant (8×8 DCT on 32×32 grayscale, mean threshold). Honest placeholder for a demo — flagged in the UI caveat. The trained model embeds whatever 896 bits we hand it; choice of `H` function is independent of model behavior.
- **sig (512 bits):** Ed25519 signature of `H_bits` using a demo secret key (libsodium-wrappers). The demo keypair is generated once and hardcoded in the page source.
- **pk (256 bits):** demo public key, same place.

UI caveat near the verify badge: *"Demo keypair ships in the page; anyone can forge a valid signature. Real deployment uses device-bound secret keys."*

## Verify policy

Signature verification is **strict against the trusted demo pubkey baked into the page**, not against the `pk_bits` extracted from the recovered payload. Bit corruption in the `pk` slot does not accidentally validate.

## Attack panel (11 attacks)

All implementable in canvas. All match the training/eval set in `src/attacks.py` and `src/exp0_inn_eval.py`.

| Attack | Detail | Source |
|---|---|---|
| identity | no-op | — |
| JPEG q=80 | single pass | low tier |
| JPEG q=60 | single pass | med tier |
| JPEG q=40 | single pass | high tier (canonical-scope boundary) |
| JPEG chained 60 → 40 | two passes | `chain_p` |
| Resize ↓0.75↑ | down-up bilinear | med tier |
| Resize ↓0.5↑ | down-up bilinear | high tier |
| chain_insta | resize 0.5 → JPEG q=85 | Instagram main feed |
| chain_x | resize 0.4 → JPEG q=75 | X / Twitter post-2022 |
| chain_whatsapp_std | resize 0.5 → JPEG q=70 | WhatsApp / Facebook |
| chain_wechat | resize 0.3 → JPEG q=50 | WeChat (aggressive) |

Implementation:
- **JPEG round-trip:** `canvas.toBlob('image/jpeg', q/100)` → re-decode via `<img>` and draw back to a canvas.
- **Down-up resize:** `drawImage(src, 0, 0, scaledW, scaledH)` to a smaller canvas, then `drawImage` back up to the cropped dims. Matches the bilinear `F.interpolate` used in training.
- **Platform chains:** resize first, JPEG second — matches the training-time order ("resize first, then JPEG-encode") from `attacks.py`.

Out of scope (per model card and `attacks.py`): crop, rotate, Gaussian noise, Gaussian blur. Stated explicitly in the page copy.

## Metrics

**One-shot stats** (shown above the table, computed once per upload):
```
Image: {H} × {W} → cropped to {cropH} × {cropW} for encoding ({trimmed_px} px trimmed)
Payload: 896 bits (128 H | 512 sig | 256 pk)
Encode: {ms} ms · Decode: {ms} ms (single forward pass)
```

**Per-attack table**, with every metric measured between **attacked_container** and **attacked_cover** (both run through the same attack pipeline, so the attack damage cancels and the metric isolates the watermark's marginal perturbation under the channel):

| Attack | PSNR | SSIM | Bit accuracy | Signature |
|---|---:|---:|---:|:---:|
| identity | … | … | … | ✓ / ✗ |
| … | … | … | … | … |

- **PSNR:** `PSNR(attacked_container_crop, attacked_cover_crop)`.
- **SSIM:** Wang et al. 2004 with 11×11 Gaussian window over the encoded region.
- **Bit accuracy:** `mean(recovered_bits == original_bits)` over 896 bits.
- **Signature:** `libsodium.crypto_sign_verify_detached(recovered_sig, recovered_H, demo_pk)`.

## Page integration

- **Permalink:** `/imagehide/`
- **New page file:** `_pages/imagehide.html` using `layout: archive`. Most of the body is one `<section class="imagehide">` widget with scoped CSS.
- **Links from:**
  1. `/tools/` — new `_tools/imagehide.md` entry with `link: /imagehide/` and a one-line summary.
  2. Home page **Projects** section — adjacent to AmazingHand. (Requires adjusting `_layouts/home.html`'s Projects loop to include the new `_portfolio/imagehide.md`, or — cleaner — adding a small static entry directly in the layout. Decision: add to `_portfolio/imagehide.md` to keep the Projects section data-driven.)

## Two-stage loading UX

Page must come up instantly without a 16 MB download.

```
T=0   page HTML/CSS/JS shell renders
      Upload area visible, "Load model" button visible
      Description, methodology copy, attack list — all readable
      No fetch of onnx / libsodium yet

T=user-click "Load model"  OR  T=user drops an image
      Lazy-fetch encoder.onnx + decoder.onnx (parallel)
      Lazy-fetch libsodium-wrappers
      Progress bar: "Loading model — 4.2 / 8.1 MB"
      Detect WebGPU; fall back to WASM if unavailable, surface "WASM mode — expect 5–10× slower"

T=ready
      "Run" button becomes active

User: clicks Run
      1. Encode → cover/container/residual×10 painted, one-shot stats filled
      2. For each enabled attack (rows go queued → running → done):
         attack(container), attack(cover), decode, verify, compute metrics, fill row
```

UI states surfaced to user: `idle`, `loading-model`, `ready`, `running-encode`, `running-attacks`, `error` (with Retry).

## Aesthetic boundaries

- **Outside the widget:** academic plain shell stays (Georgia, web-blue links, masthead, footer).
- **Inside the widget:** minimal technical chrome — thin 1 px frames around image panels, monospace metric values with tabular numerals, italic-serif column headers (matches the cluster dashboard tables), one primary button, one secondary "Load default sample" link.
- No dark theme. No web fonts beyond what the site already uses (none).

## Code layout (new files on the academic site)

```
_pages/imagehide.html             ~ 150 LOC   page structure + widget HTML
_tools/imagehide.md               ~ 10 LOC    /tools/ catalog entry
_portfolio/imagehide.md           ~ 10 LOC    Projects section entry on home

assets/imagehide/
  app.js                          ~ 350 LOC   orchestration, lazy load, UI state
  pipeline.js                     ~ 200 LOC   encode + decode + ONNX session mgmt
  attacks.js                      ~ 200 LOC   JPEG / chained-JPEG / resize / 4 platform chains
  metrics.js                      ~ 100 LOC   PSNR, SSIM (11×11 gaussian window)
  payload.js                      ~ 100 LOC   pHash, Ed25519 sign/verify via libsodium
  trim.js                         ~ 60  LOC   center-trim to multiple-of-64 + paste-back
  style.css                       ~ 150 LOC   scoped widget styles
  encoder.onnx                    ~ 8  MB     INN forward
  decoder.onnx                    ~ 8  MB     INN reverse, z=0 baked in
  sample-cover.jpg                ~ 200 KB    "Load default sample" target
```

Total app code: ~1100 LOC. Models + libsodium: ~17 MB initial lazy download.

CDN dependencies (lazy):
- `onnxruntime-web` — ESM build from a pinned version on jsDelivr
- `libsodium-wrappers` — ESM build from jsDelivr

## ONNX export (model-repo side)

New file in the model repo:

`/work/pi_nwycoff_umass_edu/hang/imagehide/scripts/export_onnx.py`

Loads `results/exp0_inn_idea084_d001_full/ckpt_best.pt`, constructs the `INNCodec` model from `src/inn_model.py`, runs `torch.onnx.export` twice:

1. **Encoder graph** — wraps `model.embed(host_rgb, bits)`. Inputs: `host_rgb` shape `(1, 3, H, W)` float in `[-1, 1]`, `bits` shape `(1, 896)` float in `{0, 1}`. Output: `container_rgb` shape `(1, 3, H, W)` float in `[-1, 1]`. Dynamic axes: `{H: "H", W: "W"}` on `host_rgb` and `container_rgb`.

2. **Decoder graph** — wraps `model.extract(container_rgb)`. Input: `container_rgb` shape `(1, 3, H, W)` float in `[-1, 1]`. Output: `logits` shape `(1, 896)` float. Same dynamic spatial dims.

Both export with `opset_version=17` (supported by ONNX Runtime Web ≥ 1.17), `dynamo=False` (the classic exporter handles the model's control flow fine; the dynamo path may not yet handle the in-place DKiS coupling ops).

Validation step: load the exported `.onnx` via `onnxruntime` (Python), run on a fixed `(1, 3, 64, 64)` input, compare numerically against the PyTorch forward — `max abs error < 1e-4` required to consider the export successful.

Output: two files in `scripts/onnx_out/`, then copied by hand into `assets/imagehide/` of the academic site.

**Risks to flag for the export task:**
- The `_INNBackbone` iterates blocks in a Python `for` loop with a runtime `rev` boolean. `torch.onnx.export` traces a fixed-direction path, which is fine since we export the forward and reverse passes as **separate graphs**. No control flow needs to survive into ONNX.
- DKiS coupling uses element-wise ops + standard convolutions — all natively supported in ONNX opset 17.
- `make_dkis_keys` generates fixed scrambling keys cached on `_dummy.device`; the keys must be baked into the exported graph as constants (they are tensors, not parameters, so they may need to be moved into the graph manually if `onnx.export` doesn't capture them — verify in the validation step).
- Haar DWT/iDWT are fixed convolutions with hardcoded filters — should export cleanly.

If export fails on the Unity HPC node (no Python with PyTorch in the academic-site context, but the model repo on Unity has its `.venv`), the export runs there and produces artifacts to copy. The export script does not need to run from inside the academic-site build.

## Risks & open questions

1. **WebGPU coverage.** ONNX Runtime Web's WebGPU backend is solid in recent Chrome/Edge but limited in Safari/Firefox. WASM fallback works everywhere but is 5–10× slower for this model. Risk is acceptable; UI surfaces the mode explicitly.
2. **Model artifact size.** Default fp32 export is ~16 MB per graph. The plan ships fp16 weights (~8 MB per graph) via ONNX's `onnxconverter-common` quantization helper. If fp16 hurts bit accuracy, fall back to fp32 and accept the larger download.
3. **Mobile.** 17 MB total + WebGPU not universal → degraded mobile experience. Page is desktop-first; mobile users get an upfront notice + can still use WASM.
4. **First-inference cold start.** WebGPU shader compilation takes 1–3 s on first forward. Surface this as a "warming up" message on the first Run click.
5. **Cross-repo deliverable.** The export script lives in `imagehide`, the page lives in `yanghangAI.github.io`. The `.onnx` files are copied manually between repos. Acceptable for now; if the export gets re-run often, a small `Makefile` rule or a CI step that pulls from a release artifact is a future improvement (out of scope).
6. **pHash mismatch with future "real" `H`.** The demo pHash is not the H function the MWIP paper will eventually standardize on. Flagged in UI; not a build risk.

## Out of scope

- Real provenance H (production-grade robust hash, e.g., NeuralHash-style)
- Crop, rotate, Gaussian noise, Gaussian blur attacks
- Tamper localization / partial verification
- Adversarial attacks on H (the central research risk for the paper itself)
- Storing or sharing watermarked images
- Streaming inference / progress callbacks during a single forward pass
- Mobile-optimized UI
- Multi-image batch encoding
- A way for users to provide their own keypair
