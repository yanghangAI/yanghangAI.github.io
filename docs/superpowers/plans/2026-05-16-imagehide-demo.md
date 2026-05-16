# Imagehide Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an interactive in-browser demo at `https://yanghangAI.github.io/imagehide/` that lets a user upload any photo and watch the MWIP INN watermark embed, survive 11 channel attacks, and verify per attack — per `docs/superpowers/specs/2026-05-16-imagehide-demo-design.md`.

**Architecture:** ONNX Runtime Web runs two ONNX graphs (encoder, decoder) exported from the trained PyTorch INN. Browser canvas implements JPEG/resize/platform-chain attacks. libsodium-wrappers handles Ed25519. Page lives at `/imagehide/` on the academic site; uses a two-stage lazy load (UI first, model on first interaction). Spans two repos: ONNX export script in `/work/pi_nwycoff_umass_edu/hang/imagehide`, everything else in `/home/hangyang_umass_edu/yanghangAI.github.io`.

**Tech Stack:** Python 3 / PyTorch (export-only); plain HTML + Liquid (Jekyll); vanilla ES modules; ONNX Runtime Web (WebGPU + WASM); libsodium-wrappers; Node ≥18 built-in `--test` runner for unit tests on pure-function modules.

**Engineer notes:**
- Cross-repo. The ONNX export task (Task 1, 2) runs in `/work/pi_nwycoff_umass_edu/hang/imagehide/`. All other tasks run in `/home/hangyang_umass_edu/yanghangAI.github.io/`. Each task header states the working directory.
- This is a SLURM HPC cluster. Compute work (including the ONNX export) goes through `srun`/`sbatch`, not direct shell.
- The conda env at `/work/pi_nwycoff_umass_edu/.conda/envs/hang/` has PyTorch but NOT `onnx`/`onnxruntime`. Task 1 installs them.
- No local Jekyll on this node — final Pages build verification happens via GitHub Actions after push (Task 13).
- Node ≥18 is required for `node --test` (verified: `v25.7.0` available).
- Tests for pure functions (`trim`, `metrics`, `payload`) run under `node --test tests/imagehide/`. Browser-API modules (`attacks`, `pipeline`, `app`) are verified by loading the live page.
- Commit after every task. Push only at Task 13 unless explicitly noted.

---

## Task 1: ONNX export script (model-repo side)

**Working directory:** `/work/pi_nwycoff_umass_edu/hang/imagehide`

**Files:**
- Create: `scripts/export_onnx.py`
- Create: `scripts/export_onnx.sbatch`

- [ ] **Step 1: Confirm dependencies in the conda env**

```bash
PY=/work/pi_nwycoff_umass_edu/.conda/envs/hang/bin/python
$PY -c "import torch; print(torch.__version__)"
$PY -c "import onnx, onnxruntime" 2>&1 | tail
```
If `onnx`/`onnxruntime` are missing, install:
```bash
$PY -m pip install onnx onnxruntime
```
Expected: subsequent `$PY -c "import onnx, onnxruntime; print(onnx.__version__, onnxruntime.__version__)"` prints two version strings.

- [ ] **Step 2: Inspect the model entry points**

Read `src/inn_model.py` lines 234–333 to confirm `INNCodec.embed(host_rgb, bits)` and `INNCodec.extract(container_rgb)` are the export surfaces. Note: input pixel range `[-1, 1]`, spatial dims must be `% 64 == 0`, bits are float in `{0, 1}` shape `(B, 896)`.

- [ ] **Step 3: Write `scripts/export_onnx.py`**

```python
#!/usr/bin/env python
"""Export the INNCodec at results/exp0_inn_idea084_d001_full/ckpt_best.pt
to two separate ONNX graphs: encoder (embed) and decoder (extract).

Validates each export numerically against the PyTorch reference on a
fixed (1, 3, 64, 64) input. Max abs error must be < 1e-4 to pass.

Outputs:
  scripts/onnx_out/encoder.onnx
  scripts/onnx_out/decoder.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

# Make the project's src/ importable when running from repo root.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.inn_model import INNCodec  # noqa: E402


CKPT_DEFAULT = REPO / "results/exp0_inn_idea084_d001_full/ckpt_best.pt"
OUT_DIR = Path(__file__).parent / "onnx_out"
N_BITS = 896
PROBE_HW = 64  # divisible by 64; small enough for fast numerical check


class EncoderWrap(torch.nn.Module):
    """Wraps INNCodec.embed for ONNX export — single tensor-in/tensor-out."""
    def __init__(self, codec: INNCodec):
        super().__init__()
        self.codec = codec

    def forward(self, host_rgb: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        return self.codec.embed(host_rgb, bits)


class DecoderWrap(torch.nn.Module):
    """Wraps INNCodec.extract (z=0 baked in)."""
    def __init__(self, codec: INNCodec):
        super().__init__()
        self.codec = codec

    def forward(self, container_rgb: torch.Tensor) -> torch.Tensor:
        return self.codec.extract(container_rgb)


def load_codec(ckpt_path: Path) -> INNCodec:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Trainer typically saves {"model": state_dict, ...}. Handle both.
    sd = state.get("model", state) if isinstance(state, dict) else state
    codec = INNCodec()  # defaults match the training config
    codec.load_state_dict(sd)
    codec.eval()
    return codec


def export_one(module: torch.nn.Module, args: tuple, out_path: Path,
               input_names: list[str], output_names: list[str],
               dynamic_axes: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        args,
        out_path.as_posix(),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(out_path.as_posix()))


def validate_numerical(out_path: Path, pt_module: torch.nn.Module,
                       args: tuple, tol: float = 1e-4) -> float:
    """Compare PyTorch reference forward vs onnxruntime forward."""
    with torch.no_grad():
        ref = pt_module(*args)
    sess = ort.InferenceSession(out_path.as_posix(),
                                providers=["CPUExecutionProvider"])
    feeds = {name: a.cpu().numpy() for name, a in zip(
        [i.name for i in sess.get_inputs()], args)}
    out = sess.run(None, feeds)[0]
    err = float(np.max(np.abs(ref.cpu().numpy() - out)))
    print(f"  {out_path.name}: max abs error = {err:.6g}")
    if err > tol:
        raise SystemExit(f"FAIL: {out_path.name} exceeds tolerance {tol}")
    return err


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT_DEFAULT))
    args_cli = p.parse_args()

    print(f"loading {args_cli.ckpt}")
    codec = load_codec(Path(args_cli.ckpt))
    print(f"  params: {sum(p.numel() for p in codec.parameters()):,}")

    encoder = EncoderWrap(codec).eval()
    decoder = DecoderWrap(codec).eval()

    probe_img = torch.rand(1, 3, PROBE_HW, PROBE_HW) * 2 - 1  # [-1, 1]
    probe_bits = (torch.rand(1, N_BITS) > 0.5).float()

    print("exporting encoder...")
    enc_path = OUT_DIR / "encoder.onnx"
    export_one(
        encoder,
        (probe_img, probe_bits),
        enc_path,
        input_names=["host_rgb", "bits"],
        output_names=["container_rgb"],
        dynamic_axes={
            "host_rgb":     {2: "H", 3: "W"},
            "container_rgb":{2: "H", 3: "W"},
        },
    )
    validate_numerical(enc_path, encoder, (probe_img, probe_bits))

    print("exporting decoder...")
    dec_path = OUT_DIR / "decoder.onnx"
    # Use the actual container from the encoder, not random pixels — keeps the
    # decoder reference in a realistic input regime.
    with torch.no_grad():
        container = codec.embed(probe_img, probe_bits)
    export_one(
        decoder,
        (container,),
        dec_path,
        input_names=["container_rgb"],
        output_names=["bit_logits"],
        dynamic_axes={
            "container_rgb": {2: "H", 3: "W"},
            "bit_logits":    {0: "B"},
        },
    )
    validate_numerical(dec_path, decoder, (container,))

    enc_mb = enc_path.stat().st_size / 1e6
    dec_mb = dec_path.stat().st_size / 1e6
    print(f"done. encoder.onnx {enc_mb:.1f} MB, decoder.onnx {dec_mb:.1f} MB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the SLURM wrapper `scripts/export_onnx.sbatch`**

```bash
#!/bin/bash
#SBATCH -J imagehide-export-onnx
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 0:20:00
#SBATCH -o logs/export_onnx_%j.out
set -euo pipefail
PY=/work/pi_nwycoff_umass_edu/.conda/envs/hang/bin/python
cd /work/pi_nwycoff_umass_edu/hang/imagehide
mkdir -p logs
$PY scripts/export_onnx.py
ls -lh scripts/onnx_out/
```

- [ ] **Step 5: Commit**

```bash
cd /work/pi_nwycoff_umass_edu/hang/imagehide
git add scripts/export_onnx.py scripts/export_onnx.sbatch
git commit -m "feat(export): ONNX export script for encoder/decoder graphs"
```

---

## Task 2: Run the export, validate, copy artifacts to the site repo

**Working directory:** start in `/work/pi_nwycoff_umass_edu/hang/imagehide`, finish in `/home/hangyang_umass_edu/yanghangAI.github.io`

**Files:**
- Produced by export: `scripts/onnx_out/encoder.onnx`, `scripts/onnx_out/decoder.onnx`
- Copied to: `/home/hangyang_umass_edu/yanghangAI.github.io/assets/imagehide/encoder.onnx`, `decoder.onnx`
- Modify: `.gitignore` in academic site repo (allow committing `.onnx` files in `assets/imagehide/`)

- [ ] **Step 1: Submit the export job**

```bash
cd /work/pi_nwycoff_umass_edu/hang/imagehide
sbatch scripts/export_onnx.sbatch
```
Note the job ID. Wait for it to complete:
```bash
until sacct -j <JOBID> --format=State -n -X | grep -qE "COMPLETED|FAILED|CANCELLED|TIMEOUT"; do sleep 5; done
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed -X
```
Expected: `State` = `COMPLETED`, `ExitCode` = `0:0`.

- [ ] **Step 2: Inspect the validation output**

```bash
cat logs/export_onnx_<JOBID>.out
```
Expected output contains:
- `loading ...ckpt_best.pt` followed by `params: 4,0X X,XXX` (around 4 million)
- `encoder.onnx: max abs error = <number less than 1e-4>`
- `decoder.onnx: max abs error = <number less than 1e-4>`
- final line listing both files with their sizes (each ~8–16 MB)

If either error exceeds tolerance, STOP — escalate. Possible causes: opset issues with a custom op, fp32 vs fp64 drift, missing constants. Do NOT proceed to copy.

- [ ] **Step 3: Copy artifacts to the academic site repo**

```bash
mkdir -p /home/hangyang_umass_edu/yanghangAI.github.io/assets/imagehide
cp /work/pi_nwycoff_umass_edu/hang/imagehide/scripts/onnx_out/encoder.onnx \
   /home/hangyang_umass_edu/yanghangAI.github.io/assets/imagehide/encoder.onnx
cp /work/pi_nwycoff_umass_edu/hang/imagehide/scripts/onnx_out/decoder.onnx \
   /home/hangyang_umass_edu/yanghangAI.github.io/assets/imagehide/decoder.onnx
ls -lh /home/hangyang_umass_edu/yanghangAI.github.io/assets/imagehide/
```

- [ ] **Step 4: Confirm `.gitignore` allows the `.onnx` files**

```bash
cd /home/hangyang_umass_edu/yanghangAI.github.io
cat .gitignore
```
If the file lists `*.onnx` or `assets/imagehide/`, edit it to either remove those entries or add `!assets/imagehide/*.onnx` as a negation. Most likely no edit needed (the existing `.gitignore` was inherited from academicpages and does not block `.onnx`).

- [ ] **Step 5: Commit**

```bash
cd /home/hangyang_umass_edu/yanghangAI.github.io
git add assets/imagehide/encoder.onnx assets/imagehide/decoder.onnx
git commit -m "feat(imagehide): add exported ONNX models (encoder + decoder)"
```
**Do NOT push yet** — files are large (~16 MB total). Pushing happens with the rest in Task 13.

---

## Task 3: `trim.js` — center-trim and paste-back, with tests

**Working directory:** `/home/hangyang_umass_edu/yanghangAI.github.io`

**Files:**
- Create: `assets/imagehide/trim.js`
- Create: `tests/imagehide/trim.test.js`
- Modify: `_config.yml` (add `tests/` to `exclude:`)

- [ ] **Step 1: Add `tests/` to Jekyll's exclude list**

In `_config.yml`, find the `exclude:` block (around line 70 in the recently rewritten config). Append `tests/` to the list. Example:

```yaml
exclude:
  - Gemfile
  - Gemfile.lock
  - LICENSE
  - README.md
  - docs/
  - scripts/
  - tests/
  - vendor/
  - node_modules/
```

- [ ] **Step 2: Write the failing test**

Create `tests/imagehide/trim.test.js`:

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { computeCrop, splitTrim, pasteBack } from '../../assets/imagehide/trim.js';

test('computeCrop: already-aligned dims pass through unchanged', () => {
  const c = computeCrop(1024, 1024);
  assert.deepEqual(c, { cropH: 1024, cropW: 1024, top: 0, left: 0,
                        trimmedTop: 0, trimmedBottom: 0,
                        trimmedLeft: 0, trimmedRight: 0 });
});

test('computeCrop: 1920x1080 trims to 1920x1024 (center)', () => {
  const c = computeCrop(1080, 1920);
  assert.equal(c.cropH, 1024);   // 1080 - (1080 % 64) = 1080 - 56 = 1024
  assert.equal(c.cropW, 1920);   // 1920 % 64 = 0
  assert.equal(c.top, 28);       // 56 / 2 = 28
  assert.equal(c.left, 0);
  assert.equal(c.trimmedTop, 28);
  assert.equal(c.trimmedBottom, 28);
});

test('computeCrop: odd-trim splits with extra pixel on the bottom/right', () => {
  // 1023 - (1023 % 64) = 1023 - 63 = 960; trim = 63; split = 31 top / 32 bottom
  const c = computeCrop(1023, 1023);
  assert.equal(c.cropH, 960);
  assert.equal(c.trimmedTop, 31);
  assert.equal(c.trimmedBottom, 32);
});

test('splitTrim then pasteBack reconstructs the original RGBA bytes', () => {
  // Construct a 128x128 RGBA ImageData-like object with a deterministic pattern.
  const H = 128, W = 128;
  const data = new Uint8ClampedArray(H * W * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = (i / 4) % 256;
    data[i + 1] = ((i / 4) >> 8) % 256;
    data[i + 2] = ((i / 4) >> 4) % 256;
    data[i + 3] = 255;
  }
  const original = { data, width: W, height: H };
  // computeCrop says 128 → 128 (already aligned), but force a synthetic crop
  // by passing a smaller (cropH, cropW) to splitTrim directly.
  const crop = { cropH: 64, cropW: 64, top: 32, left: 32,
                 trimmedTop: 32, trimmedBottom: 32,
                 trimmedLeft: 32, trimmedRight: 32 };
  const { core, strips } = splitTrim(original, crop);
  assert.equal(core.width, 64);
  assert.equal(core.height, 64);
  const recon = pasteBack(core, strips, crop, W, H);
  assert.equal(recon.width, W);
  assert.equal(recon.height, H);
  assert.deepEqual(recon.data, original.data);
});
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd /home/hangyang_umass_edu/yanghangAI.github.io
node --test tests/imagehide/trim.test.js 2>&1 | head -20
```
Expected: failures with "Cannot find module" or "is not a function".

- [ ] **Step 4: Implement `assets/imagehide/trim.js`**

```javascript
/**
 * Center-trim helpers for the imagehide demo.
 *
 * Browser ImageData (`{ data: Uint8ClampedArray, width, height }`) is the
 * input format. Bytes are interleaved RGBA, row-major.
 *
 * No DOM dependency — these functions take plain objects and work in Node tests.
 */

const ALIGN = 64;  // INN requires H % 64 == 0 and W % 64 == 0

export function computeCrop(H, W) {
  const cropH = H - (H % ALIGN);
  const cropW = W - (W % ALIGN);
  const trimH = H - cropH;
  const trimW = W - cropW;
  // Center the crop; odd trim puts the extra pixel on bottom/right.
  const trimmedTop = trimH >> 1;
  const trimmedBottom = trimH - trimmedTop;
  const trimmedLeft = trimW >> 1;
  const trimmedRight = trimW - trimmedLeft;
  return {
    cropH, cropW,
    top: trimmedTop, left: trimmedLeft,
    trimmedTop, trimmedBottom, trimmedLeft, trimmedRight,
  };
}

export function splitTrim(imageData, crop) {
  const { data: src, width: W, height: H } = imageData;
  const { cropH, cropW, top, left } = crop;
  const core = new Uint8ClampedArray(cropH * cropW * 4);
  for (let y = 0; y < cropH; y++) {
    const srcRow = ((top + y) * W + left) * 4;
    const dstRow = y * cropW * 4;
    core.set(src.subarray(srcRow, srcRow + cropW * 4), dstRow);
  }
  // Strips are the original full-size frame; pasteBack overwrites only the core.
  const strips = new Uint8ClampedArray(src);
  return {
    core: { data: core, width: cropW, height: cropH },
    strips,
  };
}

export function pasteBack(coreImageData, strips, crop, fullW, fullH) {
  const { data: core, width: cropW, height: cropH } = coreImageData;
  const { top, left } = crop;
  const out = new Uint8ClampedArray(strips);  // start from original strips
  for (let y = 0; y < cropH; y++) {
    const srcRow = y * cropW * 4;
    const dstRow = ((top + y) * fullW + left) * 4;
    out.set(core.subarray(srcRow, srcRow + cropW * 4), dstRow);
  }
  return { data: out, width: fullW, height: fullH };
}
```

- [ ] **Step 5: Run tests, expect pass**

```bash
node --test tests/imagehide/trim.test.js 2>&1 | tail -10
```
Expected: `# pass 4`, `# fail 0`.

- [ ] **Step 6: Commit**

```bash
git add _config.yml assets/imagehide/trim.js tests/imagehide/trim.test.js
git commit -m "feat(imagehide): center-trim utilities with tests"
```

---

## Task 4: `payload.js` — bits ↔ bytes, pHash, Ed25519 wrapper, with tests

**Working directory:** `/home/hangyang_umass_edu/yanghangAI.github.io`

**Files:**
- Create: `assets/imagehide/payload.js`
- Create: `tests/imagehide/payload.test.js`

- [ ] **Step 1: Write the failing test**

Create `tests/imagehide/payload.test.js`:

```javascript
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
node --test tests/imagehide/payload.test.js 2>&1 | head -20
```
Expected: import failures.

- [ ] **Step 3: Implement `assets/imagehide/payload.js`**

```javascript
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
```

- [ ] **Step 4: Run tests, expect pass**

```bash
node --test tests/imagehide/payload.test.js 2>&1 | tail -10
```
Expected: `# pass 6`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add assets/imagehide/payload.js tests/imagehide/payload.test.js
git commit -m "feat(imagehide): payload helpers (bits/bytes, packing, pHash) with tests"
```

---

## Task 5: `metrics.js` — PSNR + SSIM with tests

**Working directory:** `/home/hangyang_umass_edu/yanghangAI.github.io`

**Files:**
- Create: `assets/imagehide/metrics.js`
- Create: `tests/imagehide/metrics.test.js`

- [ ] **Step 1: Write the failing test**

Create `tests/imagehide/metrics.test.js`:

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { psnr, ssim } from '../../assets/imagehide/metrics.js';

function makeRgba(w, h, fill) {
  const d = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < d.length; i += 4) {
    d[i] = fill; d[i+1] = fill; d[i+2] = fill; d[i+3] = 255;
  }
  return { data: d, width: w, height: h };
}

test('psnr: identical images → Infinity', () => {
  const a = makeRgba(32, 32, 128);
  const b = makeRgba(32, 32, 128);
  assert.equal(psnr(a, b), Infinity);
});

test('psnr: known MSE → known PSNR', () => {
  // All-zero vs all-1 over RGB (alpha skipped): MSE = 1, PSNR = 20*log10(255) ≈ 48.13
  const a = makeRgba(8, 8, 0);
  const b = makeRgba(8, 8, 1);
  const v = psnr(a, b);
  assert.ok(Math.abs(v - 48.131) < 0.01, `got ${v}`);
});

test('ssim: identical images → 1.0', () => {
  const a = makeRgba(64, 64, 100);
  const b = makeRgba(64, 64, 100);
  const v = ssim(a, b);
  assert.ok(Math.abs(v - 1.0) < 1e-6, `got ${v}`);
});

test('ssim: very different images → near 0 or low', () => {
  const a = makeRgba(64, 64, 0);
  const b = makeRgba(64, 64, 255);
  const v = ssim(a, b);
  assert.ok(v < 0.1, `expected very low SSIM for max contrast, got ${v}`);
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
node --test tests/imagehide/metrics.test.js 2>&1 | head
```
Expected: import errors.

- [ ] **Step 3: Implement `assets/imagehide/metrics.js`**

```javascript
/**
 * PSNR and SSIM on RGBA ImageData. RGB channels only; alpha ignored.
 *
 * Both images must have the same dimensions.
 *
 * SSIM: Wang et al. 2004 with 11×11 Gaussian window, σ=1.5, K1=0.01, K2=0.03,
 * L=255. Computed per-channel on RGB, averaged across channels. The window
 * is applied at unit stride; the SSIM map is averaged over valid positions.
 */

const K1 = 0.01, K2 = 0.03, L = 255;
const C1 = (K1 * L) ** 2;
const C2 = (K2 * L) ** 2;

export function psnr(aImg, bImg) {
  if (aImg.width !== bImg.width || aImg.height !== bImg.height) {
    throw new Error('PSNR: dimension mismatch');
  }
  const a = aImg.data, b = bImg.data;
  let sse = 0, n = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const d = a[i + c] - b[i + c];
      sse += d * d;
      n++;
    }
  }
  if (sse === 0) return Infinity;
  const mse = sse / n;
  return 10 * Math.log10((L * L) / mse);
}

function gaussianKernel1D(size, sigma) {
  const k = new Float64Array(size);
  const half = (size - 1) / 2;
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const x = i - half;
    k[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += k[i];
  }
  for (let i = 0; i < size; i++) k[i] /= sum;
  return k;
}

function convolve2DSeparable(src, W, H, k1d) {
  const r = (k1d.length - 1) / 2;
  const tmp = new Float64Array(W * H);
  // Horizontal pass
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let dx = -r; dx <= r; dx++) {
        const sx = Math.min(W - 1, Math.max(0, x + dx));
        s += src[y * W + sx] * k1d[dx + r];
      }
      tmp[y * W + x] = s;
    }
  }
  const out = new Float64Array(W * H);
  // Vertical pass
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let dy = -r; dy <= r; dy++) {
        const sy = Math.min(H - 1, Math.max(0, y + dy));
        s += tmp[sy * W + x] * k1d[dy + r];
      }
      out[y * W + x] = s;
    }
  }
  return out;
}

function extractChannel(img, c) {
  const { data, width: W, height: H } = img;
  const out = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) out[i] = data[i * 4 + c];
  return out;
}

function ssimChannel(a, b, W, H, kernel) {
  // Per-channel SSIM map mean.
  const muA = convolve2DSeparable(a, W, H, kernel);
  const muB = convolve2DSeparable(b, W, H, kernel);
  const a2 = new Float64Array(W * H);
  const b2 = new Float64Array(W * H);
  const ab = new Float64Array(W * H);
  for (let i = 0; i < a.length; i++) {
    a2[i] = a[i] * a[i];
    b2[i] = b[i] * b[i];
    ab[i] = a[i] * b[i];
  }
  const muA2 = convolve2DSeparable(a2, W, H, kernel);
  const muB2 = convolve2DSeparable(b2, W, H, kernel);
  const muAB = convolve2DSeparable(ab, W, H, kernel);

  let sum = 0;
  const N = W * H;
  for (let i = 0; i < N; i++) {
    const mA = muA[i], mB = muB[i];
    const sigA2 = muA2[i] - mA * mA;
    const sigB2 = muB2[i] - mB * mB;
    const sigAB = muAB[i] - mA * mB;
    const num = (2 * mA * mB + C1) * (2 * sigAB + C2);
    const den = (mA * mA + mB * mB + C1) * (sigA2 + sigB2 + C2);
    sum += num / den;
  }
  return sum / N;
}

export function ssim(aImg, bImg) {
  if (aImg.width !== bImg.width || aImg.height !== bImg.height) {
    throw new Error('SSIM: dimension mismatch');
  }
  const W = aImg.width, H = aImg.height;
  const kernel = gaussianKernel1D(11, 1.5);
  let acc = 0;
  for (let c = 0; c < 3; c++) {
    acc += ssimChannel(extractChannel(aImg, c), extractChannel(bImg, c),
                       W, H, kernel);
  }
  return acc / 3;
}
```

- [ ] **Step 4: Run tests, expect pass**

```bash
node --test tests/imagehide/metrics.test.js 2>&1 | tail -10
```
Expected: `# pass 4`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add assets/imagehide/metrics.js tests/imagehide/metrics.test.js
git commit -m "feat(imagehide): PSNR + SSIM (11x11 gaussian) with tests"
```

---

## Task 6: `attacks.js` — JPEG, chained-JPEG, resize, 4 platform chains

Browser-only (uses Canvas API and `Blob`/`createImageBitmap`); no Node tests. Verified by manual browser smoke test in Task 13.

**Working directory:** `/home/hangyang_umass_edu/yanghangAI.github.io`

**Files:**
- Create: `assets/imagehide/attacks.js`

- [ ] **Step 1: Write the module**

```javascript
/**
 * Channel attacks for the imagehide demo. All operate on ImageData
 * `{ data: Uint8ClampedArray, width, height }`. RGBA in/out.
 *
 * Attacks:
 *   identity            no-op
 *   jpeg(q)             single JPEG round-trip at quality q (0–100)
 *   jpegChain(q1, q2)   two JPEG passes, q1 first then q2
 *   resize(scale)       bilinear down to scale×scale, then up to original
 *   chain(scale, q)     platform pipeline: resize then JPEG
 */

async function imageDataToBlob(img, type, quality) {
  const c = new OffscreenCanvas(img.width, img.height);
  c.getContext('2d').putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
  return c.convertToBlob({ type, quality });
}

async function blobToImageData(blob) {
  const bitmap = await createImageBitmap(blob);
  const c = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = c.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, bitmap.width, bitmap.height);
}

export async function identity(img) {
  return new ImageData(new Uint8ClampedArray(img.data), img.width, img.height);
}

export async function jpeg(img, q) {
  const blob = await imageDataToBlob(img, 'image/jpeg', q / 100);
  return blobToImageData(blob);
}

export async function jpegChain(img, q1, q2) {
  const once = await jpeg(img, q1);
  return jpeg(once, q2);
}

export async function resize(img, scale) {
  const W = img.width, H = img.height;
  const sw = Math.max(1, Math.round(W * scale));
  const sh = Math.max(1, Math.round(H * scale));
  const small = new OffscreenCanvas(sw, sh);
  const sctx = small.getContext('2d');
  sctx.imageSmoothingEnabled = true;
  sctx.imageSmoothingQuality = 'medium';   // 'medium' approximates bilinear
  const srcCanvas = new OffscreenCanvas(W, H);
  srcCanvas.getContext('2d').putImageData(
    new ImageData(img.data, W, H), 0, 0);
  sctx.drawImage(srcCanvas, 0, 0, sw, sh);
  const big = new OffscreenCanvas(W, H);
  const bctx = big.getContext('2d');
  bctx.imageSmoothingEnabled = true;
  bctx.imageSmoothingQuality = 'medium';
  bctx.drawImage(small, 0, 0, W, H);
  return bctx.getImageData(0, 0, W, H);
}

export async function chain(img, scale, q) {
  const resized = await resize(img, scale);
  return jpeg(resized, q);
}

// Catalog used by the UI. Keep keys stable — they appear as row IDs.
export const ATTACKS = [
  { id: 'identity',            label: 'identity',            fn: (img) => identity(img) },
  { id: 'jpeg_q80',            label: 'JPEG q=80',           fn: (img) => jpeg(img, 80) },
  { id: 'jpeg_q60',            label: 'JPEG q=60',           fn: (img) => jpeg(img, 60) },
  { id: 'jpeg_q40',            label: 'JPEG q=40',           fn: (img) => jpeg(img, 40) },
  { id: 'jpeg_chain_60_40',    label: 'JPEG 60 → 40',        fn: (img) => jpegChain(img, 60, 40) },
  { id: 'resize_075',          label: 'Resize ↓0.75↑',       fn: (img) => resize(img, 0.75) },
  { id: 'resize_050',          label: 'Resize ↓0.5↑',        fn: (img) => resize(img, 0.5) },
  { id: 'chain_insta',         label: 'chain_insta',         fn: (img) => chain(img, 0.5, 85) },
  { id: 'chain_x',             label: 'chain_x',             fn: (img) => chain(img, 0.4, 75) },
  { id: 'chain_whatsapp_std',  label: 'chain_whatsapp_std',  fn: (img) => chain(img, 0.5, 70) },
  { id: 'chain_wechat',        label: 'chain_wechat',        fn: (img) => chain(img, 0.3, 50) },
];
```

- [ ] **Step 2: Commit**

```bash
git add assets/imagehide/attacks.js
git commit -m "feat(imagehide): canvas-based channel attacks (JPEG, resize, 4 platforms)"
```

---

## Task 7: `pipeline.js` — ONNX session manager + encode/decode

Browser-only. Wraps ONNX Runtime Web for the two model graphs and exposes a tiny encode/decode API consumed by `app.js`.

**Working directory:** `/home/hangyang_umass_edu/yanghangAI.github.io`

**Files:**
- Create: `assets/imagehide/pipeline.js`

- [ ] **Step 1: Write the module**

```javascript
/**
 * ONNX Runtime Web wrapper for the MWIP encoder/decoder graphs.
 *
 * Encoder input:  host_rgb tensor [1, 3, H, W] float32 in [-1, 1]
 *                 bits     tensor [1, 896]     float32 in {0, 1}
 * Encoder output: container_rgb [1, 3, H, W] float32 in [-1, 1]
 *
 * Decoder input:  container_rgb [1, 3, H, W] float32 in [-1, 1]
 * Decoder output: bit_logits   [1, 896]    float32 (apply sigmoid + 0.5 threshold)
 *
 * Backend: WebGPU when available, falls back to WASM. Surface the active
 * backend via getBackend() so the UI can warn users on the slow path.
 */

let ort = null;          // loaded lazily
let encoderSession = null;
let decoderSession = null;
let activeBackend = null;
let loadPromise = null;

const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.webgpu.min.mjs';

async function detectWebGPU() {
  if (!('gpu' in navigator)) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch { return false; }
}

export function getBackend() { return activeBackend; }

export async function loadModels(encoderUrl, decoderUrl, onProgress) {
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    if (!ort) ort = await import(/* @vite-ignore */ ORT_CDN);
    activeBackend = (await detectWebGPU()) ? 'webgpu' : 'wasm';
    // Fetch both .onnx files with a shared progress callback.
    const [encBuf, decBuf] = await Promise.all([
      fetchWithProgress(encoderUrl, onProgress, 'encoder'),
      fetchWithProgress(decoderUrl, onProgress, 'decoder'),
    ]);
    const sessionOpts = { executionProviders: [activeBackend] };
    encoderSession = await ort.InferenceSession.create(encBuf, sessionOpts);
    decoderSession = await ort.InferenceSession.create(decBuf, sessionOpts);
  })();
  return loadPromise;
}

async function fetchWithProgress(url, cb, tag) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url} → ${resp.status}`);
  const total = Number(resp.headers.get('content-length')) || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (cb) cb({ tag, loaded, total });
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf.buffer;
}

// ---------- tensor helpers ----------

function imageDataToFloat32CHW(imageData) {
  // RGBA [0..255] → CHW [-1..1] float32
  const { data, width: W, height: H } = imageData;
  const out = new Float32Array(3 * H * W);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const pi = y * W + x;
      out[0 * H * W + pi] = (data[i]     / 127.5) - 1;
      out[1 * H * W + pi] = (data[i + 1] / 127.5) - 1;
      out[2 * H * W + pi] = (data[i + 2] / 127.5) - 1;
    }
  }
  return out;
}

function float32CHWtoImageData(arr, W, H) {
  const data = new Uint8ClampedArray(H * W * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pi = y * W + x;
      const i = pi * 4;
      data[i]     = Math.round(Math.min(255, Math.max(0, (arr[0 * H * W + pi] + 1) * 127.5)));
      data[i + 1] = Math.round(Math.min(255, Math.max(0, (arr[1 * H * W + pi] + 1) * 127.5)));
      data[i + 2] = Math.round(Math.min(255, Math.max(0, (arr[2 * H * W + pi] + 1) * 127.5)));
      data[i + 3] = 255;
    }
  }
  return { data, width: W, height: H };
}

function bitsToFloat32(bitsUint8) {
  const out = new Float32Array(bitsUint8.length);
  for (let i = 0; i < bitsUint8.length; i++) out[i] = bitsUint8[i];
  return out;
}

// ---------- public API ----------

export async function encode(coverImageData, bitsUint8) {
  if (!encoderSession) throw new Error('encoder not loaded');
  const W = coverImageData.width, H = coverImageData.height;
  const imgArr = imageDataToFloat32CHW(coverImageData);
  const imgT = new ort.Tensor('float32', imgArr, [1, 3, H, W]);
  const bitsT = new ort.Tensor('float32', bitsToFloat32(bitsUint8), [1, bitsUint8.length]);
  const t0 = performance.now();
  const { container_rgb } = await encoderSession.run({ host_rgb: imgT, bits: bitsT });
  const dt = performance.now() - t0;
  return { container: float32CHWtoImageData(container_rgb.data, W, H), ms: dt };
}

export async function decode(containerImageData) {
  if (!decoderSession) throw new Error('decoder not loaded');
  const W = containerImageData.width, H = containerImageData.height;
  const arr = imageDataToFloat32CHW(containerImageData);
  const t = new ort.Tensor('float32', arr, [1, 3, H, W]);
  const t0 = performance.now();
  const { bit_logits } = await decoderSession.run({ container_rgb: t });
  const dt = performance.now() - t0;
  // Sigmoid + threshold 0.5
  const bits = new Uint8Array(bit_logits.data.length);
  for (let i = 0; i < bits.length; i++) {
    const s = 1 / (1 + Math.exp(-bit_logits.data[i]));
    bits[i] = s > 0.5 ? 1 : 0;
  }
  return { bits, ms: dt };
}
```

- [ ] **Step 2: Commit**

```bash
git add assets/imagehide/pipeline.js
git commit -m "feat(imagehide): ONNX Runtime Web pipeline (encode + decode)"
```

---

## Task 8: `style.css` — scoped widget styles

**Working directory:** `/home/hangyang_umass_edu/yanghangAI.github.io`

**Files:**
- Create: `assets/imagehide/style.css`

- [ ] **Step 1: Write the stylesheet**

```css
/* Imagehide demo — scoped widget styles.
   Everything is namespaced under .imagehide so it can't bleed into the rest
   of the academic site. Page-level chrome (masthead, footer) keeps the
   inherited Georgia + classic-blue look. */

.imagehide {
  margin: 1.5rem 0 0;
}

.imagehide__caveat {
  font-style: italic;
  color: #666;
  font-size: 0.95rem;
  margin: 0.8rem 0 1.2rem;
}

.imagehide__upload {
  border: 1px dashed #aaa;
  padding: 1rem;
  text-align: center;
  margin: 1rem 0;
  background: #fafaf6;
}
.imagehide__upload.is-dragover { background: #f0ead8; border-color: #555; }
.imagehide__upload input[type="file"] { font: inherit; }
.imagehide__upload-sample {
  display: inline-block; margin-top: 0.5rem; font-size: 0.95rem;
}

.imagehide__status {
  font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
  font-size: 0.85rem;
  color: #555;
  margin: 0.5rem 0 1rem;
}
.imagehide__status .progress {
  display: inline-block; width: 200px; height: 0.6rem; border: 1px solid #888;
  vertical-align: middle; margin: 0 0.4rem;
}
.imagehide__status .progress > span {
  display: block; height: 100%; background: #0645ad;
  width: 0%; transition: width 0.2s ease;
}
.imagehide__status .warn { color: #b00020; }

.imagehide__panels {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
  margin: 1rem 0;
}
.imagehide__panel {
  border: 1px solid #ccc;
  background: #fff;
}
.imagehide__panel canvas {
  display: block; width: 100%; height: auto; image-rendering: pixelated;
}
.imagehide__panel-label {
  font-family: ui-monospace, monospace;
  font-size: 0.78rem;
  text-align: center;
  padding: 0.25rem;
  color: #555;
  border-top: 1px solid #ccc;
}

.imagehide__oneshot {
  font-family: ui-monospace, monospace;
  font-size: 0.85rem;
  background: #f6f4ec;
  padding: 0.6rem 0.8rem;
  border-left: 3px solid #aaa;
  white-space: pre-wrap;
  margin: 0.5rem 0 1rem;
}

.imagehide__controls {
  display: flex; flex-wrap: wrap; gap: 0.4rem 1rem; align-items: center;
  margin: 0.8rem 0;
}
.imagehide__controls label {
  font-size: 0.92rem; cursor: pointer;
}
.imagehide__controls button {
  font: inherit;
  padding: 0.4rem 0.9rem; border: 1px solid #444; background: #fff; cursor: pointer;
}
.imagehide__controls button[disabled] { opacity: 0.5; cursor: not-allowed; }
.imagehide__controls button.primary {
  background: #0645ad; color: #fff; border-color: #0645ad;
}

.imagehide__results {
  width: 100%; border-collapse: collapse;
  font-family: ui-monospace, monospace;
  font-size: 0.85rem;
}
.imagehide__results th, .imagehide__results td {
  text-align: left; padding: 0.35rem 0.7rem 0.35rem 0;
  border-bottom: 1px solid #ddd; vertical-align: top;
  font-variant-numeric: tabular-nums;
}
.imagehide__results th {
  font-family: Georgia, serif; font-style: italic; font-weight: 400;
  font-size: 0.92rem; color: #555;
  border-bottom: 1px solid #888;
}
.imagehide__results td.num    { text-align: right; }
.imagehide__results td.sig-ok { color: #1f6b1f; font-weight: 700; }
.imagehide__results td.sig-no { color: #b00020; font-weight: 700; }
.imagehide__results tr.queued td  { color: #999; }
.imagehide__results tr.running td { color: #555; }

@media (max-width: 700px) {
  .imagehide__panels { grid-template-columns: 1fr; }
}
```

- [ ] **Step 2: Commit**

```bash
git add assets/imagehide/style.css
git commit -m "feat(imagehide): scoped widget styles"
```

---

## Task 9: `app.js` — orchestration, lazy load, UI state machine

Browser-only. Coordinates the modules, owns the UI state machine, lazy-loads ONNX + libsodium on first interaction.

**Working directory:** `/home/hangyang_umass_edu/yanghangAI.github.io`

**Files:**
- Create: `assets/imagehide/app.js`

- [ ] **Step 1: Write the orchestrator**

```javascript
import { computeCrop, splitTrim, pasteBack } from './trim.js';
import { phash128, packPayload, unpackPayload, bitAccuracy, N_H } from './payload.js';
import { psnr, ssim } from './metrics.js';
import { ATTACKS } from './attacks.js';
import { loadModels, encode, decode, getBackend } from './pipeline.js';

const LIBSODIUM_CDN = 'https://cdn.jsdelivr.net/npm/libsodium-wrappers@0.7.13/+esm';
const MAX_PIXELS_BEFORE_WARN = 6 * 1024 * 1024;   // ~6 MP

const els = {};
let sodium = null;
let demoKeypair = null;
let state = {
  status: 'idle',                 // idle | loading-model | ready | encoding | running-attacks | error
  originalImage: null,            // full ImageData
  crop: null,                     // computeCrop result
  coreCover: null,                // cropped cover ImageData
  fullCover: null,                // full reconstructed cover (for paste-back)
  fullContainer: null,            // full-size container after encode (for download)
  coreContainer: null,            // cropped container
  payloadBits: null,              // Uint8Array(896)
  payloadParts: null,             // { H, sig, pk }
  encodeMs: null,
  decodeMs: null,
  sizeOk: false,
};

document.addEventListener('DOMContentLoaded', init);

function init() {
  ['upload', 'file', 'sample', 'status', 'cover', 'container', 'residual',
   'oneshot', 'attackList', 'runBtn', 'resultsBody'].forEach(k => {
    els[k] = document.getElementById(`ih-${k}`);
  });
  els.file.addEventListener('change', e => onFile(e.target.files?.[0]));
  els.upload.addEventListener('dragover', e => {
    e.preventDefault(); els.upload.classList.add('is-dragover');
  });
  els.upload.addEventListener('dragleave', () => els.upload.classList.remove('is-dragover'));
  els.upload.addEventListener('drop', e => {
    e.preventDefault();
    els.upload.classList.remove('is-dragover');
    onFile(e.dataTransfer.files?.[0]);
  });
  els.sample.addEventListener('click', e => { e.preventDefault(); loadSample(); });
  els.runBtn.addEventListener('click', onRun);
  renderAttackList();
  setStatus('Drop an image or pick a file to begin.');
}

function renderAttackList() {
  els.attackList.innerHTML = ATTACKS.map(a =>
    `<label><input type="checkbox" value="${a.id}" checked> ${a.label}</label>`
  ).join('');
  els.resultsBody.innerHTML = ATTACKS.map(a =>
    `<tr id="row-${a.id}" class="queued">
       <td>${a.label}</td><td class="num">—</td><td class="num">—</td>
       <td class="num">—</td><td>—</td>
     </tr>`).join('');
}

function setStatus(html, isError = false) {
  els.status.innerHTML = html;
  els.status.classList.toggle('warn', isError);
}

async function onFile(file) {
  if (!file) return;
  setStatus(`Loading ${file.name}…`);
  try {
    const bitmap = await createImageBitmap(file);
    const c = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = c.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);
    const imgData = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    onImageLoaded(imgData);
  } catch (e) {
    setStatus(`Failed to load: ${e.message}`, true);
  }
}

async function loadSample() {
  setStatus('Loading sample…');
  try {
    const resp = await fetch('sample-cover.jpg');
    const blob = await resp.blob();
    const bitmap = await createImageBitmap(blob);
    const c = new OffscreenCanvas(bitmap.width, bitmap.height);
    c.getContext('2d').drawImage(bitmap, 0, 0);
    const imgData = c.getContext('2d').getImageData(0, 0, bitmap.width, bitmap.height);
    onImageLoaded(imgData);
  } catch (e) {
    setStatus(`Failed to load sample: ${e.message}`, true);
  }
}

function onImageLoaded(imageData) {
  state.originalImage = imageData;
  const W = imageData.width, H = imageData.height;
  state.crop = computeCrop(H, W);
  const pixels = W * H;
  state.sizeOk = pixels <= MAX_PIXELS_BEFORE_WARN;
  drawToCanvas(els.cover, imageData);
  setStatus(`Loaded ${W}×${H}. Cropping to ${state.crop.cropW}×${state.crop.cropH} for encoding.${
    state.sizeOk ? '' : ` <span class="warn">Image is ${(pixels / 1e6).toFixed(1)} MP — encoding may be slow or fail in your browser. <button id="ih-override">Run anyway</button></span>`
  }`);
  if (!state.sizeOk) {
    document.getElementById('ih-override').addEventListener('click', () => {
      state.sizeOk = true;
      setStatus(`Override: will run on ${(pixels / 1e6).toFixed(1)} MP.`);
      ensureLoaded();
    });
    return;
  }
  ensureLoaded();
}

function drawToCanvas(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext('2d').putImageData(
    new ImageData(imageData.data, imageData.width, imageData.height), 0, 0);
}

async function ensureLoaded() {
  if (state.status === 'ready') {
    els.runBtn.disabled = false;
    return;
  }
  if (state.status === 'loading-model') return;
  state.status = 'loading-model';
  setStatus('Loading model… 0%');
  try {
    await Promise.all([loadSodium(), loadModel()]);
    state.status = 'ready';
    els.runBtn.disabled = false;
    const backendNote = getBackend() === 'wasm'
      ? ' <span class="warn">Running on WASM (no WebGPU) — expect 5–10× slower.</span>'
      : '';
    setStatus(`Model loaded (backend: ${getBackend()}). Click Run to encode + attack.${backendNote}`);
  } catch (e) {
    state.status = 'error';
    setStatus(`Model load failed: ${e.message}`, true);
  }
}

async function loadSodium() {
  if (sodium) return;
  const mod = await import(/* @vite-ignore */ LIBSODIUM_CDN);
  sodium = mod.default || mod;
  await sodium.ready;
  demoKeypair = sodium.crypto_sign_keypair();
}

let loaded = { encoder: 0, decoder: 0, total: { encoder: 0, decoder: 0 } };
async function loadModel() {
  await loadModels(
    'encoder.onnx',
    'decoder.onnx',
    ({ tag, loaded: l, total }) => {
      loaded[tag] = l;
      if (total) loaded.total[tag] = total;
      const sum = loaded.encoder + loaded.decoder;
      const tot = loaded.total.encoder + loaded.total.decoder;
      const pct = tot ? Math.round(100 * sum / tot) : 0;
      setStatus(`Loading model… <span class="progress"><span style="width:${pct}%"></span></span> ${pct}% (${(sum/1e6).toFixed(1)} / ${(tot/1e6).toFixed(1)} MB)`);
    },
  );
}

async function onRun() {
  if (state.status !== 'ready' && state.status !== 'done') return;
  els.runBtn.disabled = true;
  state.status = 'encoding';
  setStatus('Encoding…');
  try {
    await runEncode();
    await runAttacks();
    state.status = 'done';
    els.runBtn.disabled = false;
  } catch (e) {
    state.status = 'error';
    setStatus(`Run failed: ${e.message}`, true);
    els.runBtn.disabled = false;
  }
}

async function runEncode() {
  const { core, strips } = splitTrim(state.originalImage, state.crop);
  state.coreCover = core;
  // Fully reconstructed cover = original (paste-back over same strips is a no-op
  // for the cover). Keep a reference for later attack pairs.
  state.fullCover = state.originalImage;

  // Build payload: H = pHash(core), sig = Ed25519(H), pk = demo public.
  const H_bytes = phash128(core);
  const sig = sodium.crypto_sign_detached(H_bytes, demoKeypair.privateKey);
  const pk = demoKeypair.publicKey;
  const bits = packPayload(H_bytes, sig, pk);
  state.payloadBits = bits;
  state.payloadParts = { H: H_bytes, sig, pk };

  const { container, ms } = await encode(core, bits);
  state.coreContainer = container;
  state.encodeMs = ms;

  // Reconstruct full-size container (untouched strips + encoded core).
  state.fullContainer = pasteBack(container, strips, state.crop,
                                  state.originalImage.width,
                                  state.originalImage.height);
  drawToCanvas(els.container, state.fullContainer);
  drawResidual(els.residual, core, container, 10);

  // Single decode time-probe on the clean container.
  const dec = await decode(container);
  state.decodeMs = dec.ms;

  const cropMsg = (state.crop.trimmedTop || state.crop.trimmedLeft)
    ? ` (${state.crop.trimmedTop + state.crop.trimmedBottom} px trimmed vertically, ${state.crop.trimmedLeft + state.crop.trimmedRight} px horizontally)`
    : '';
  els.oneshot.textContent =
    `Image: ${state.originalImage.width} × ${state.originalImage.height} → cropped to ${state.crop.cropW} × ${state.crop.cropH}${cropMsg}\n` +
    `Payload: 896 bits (128 H | 512 sig | 256 pk)\n` +
    `Encode: ${state.encodeMs.toFixed(1)} ms · Decode: ${state.decodeMs.toFixed(1)} ms`;
}

function drawResidual(canvas, coverCore, containerCore, scale) {
  const W = coverCore.width, H = coverCore.height;
  const data = new Uint8ClampedArray(H * W * 4);
  for (let i = 0; i < data.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const d = containerCore.data[i + c] - coverCore.data[i + c];
      data[i + c] = Math.max(0, Math.min(255, 128 + d * scale));
    }
    data[i + 3] = 255;
  }
  canvas.width = W; canvas.height = H;
  canvas.getContext('2d').putImageData(new ImageData(data, W, H), 0, 0);
}

async function runAttacks() {
  state.status = 'running-attacks';
  const enabled = new Set(
    [...els.attackList.querySelectorAll('input[type="checkbox"]:checked')]
      .map(el => el.value));
  // Reset rows
  for (const a of ATTACKS) {
    const row = document.getElementById(`row-${a.id}`);
    row.classList.toggle('queued', enabled.has(a.id));
    row.classList.toggle('running', false);
    row.querySelectorAll('td').forEach((td, idx) => { if (idx > 0) td.textContent = '—'; td.classList.remove('sig-ok', 'sig-no'); });
  }
  for (const a of ATTACKS) {
    if (!enabled.has(a.id)) continue;
    const row = document.getElementById(`row-${a.id}`);
    row.classList.remove('queued');
    row.classList.add('running');
    try {
      const attackedContainer = await a.fn(state.fullContainer);
      const attackedCover     = await a.fn(state.fullCover);
      // Crop both to the canonical region for the metric and decode.
      const aContCore = splitTrim(attackedContainer, state.crop).core;
      const aCovCore  = splitTrim(attackedCover, state.crop).core;
      const p = psnr(aContCore, aCovCore);
      const s = ssim(aContCore, aCovCore);
      const { bits: recBits } = await decode(aContCore);
      const acc = bitAccuracy(recBits, state.payloadBits);
      const { H: recH, sig: recSig } = unpackPayload(recBits);
      const sigOk = sodium.crypto_sign_verify_detached(recSig, recH, demoKeypair.publicKey);
      const tds = row.querySelectorAll('td');
      tds[1].textContent = isFinite(p) ? `${p.toFixed(1)} dB` : '∞ dB';
      tds[2].textContent = s.toFixed(3);
      tds[3].textContent = acc.toFixed(3);
      tds[4].textContent = sigOk ? '✓' : '✗';
      tds[4].classList.add(sigOk ? 'sig-ok' : 'sig-no');
    } catch (e) {
      const tds = row.querySelectorAll('td');
      tds[1].textContent = 'err';
      tds[1].title = e.message;
    }
    row.classList.remove('running');
  }
  setStatus('Done.');
}
```

- [ ] **Step 2: Commit**

```bash
git add assets/imagehide/app.js
git commit -m "feat(imagehide): app orchestrator, UI state machine, lazy loaders"
```

---

## Task 10: `_pages/imagehide.html` — page shell + widget HTML

**Working directory:** `/home/hangyang_umass_edu/yanghangAI.github.io`

**Files:**
- Create: `_pages/imagehide.html`

- [ ] **Step 1: Write the page**

```liquid
---
layout: archive
title: "Imagehide demo"
permalink: /imagehide/
lede: "Live demo of the MWIP (Minimal Whole-Image Provenance) INN watermark. Upload a photo; the model embeds an 896-bit payload (perceptual hash + Ed25519 signature + public key), then we run the container through 11 channel attacks and measure how the bits survive."
---

<link rel="stylesheet" href="{{ '/assets/imagehide/style.css' | relative_url }}">

<section class="imagehide">

  <p class="imagehide__caveat">
    Demo keypair ships in the page; anyone can forge a valid signature. Real deployment uses device-bound secret keys. The pHash used for <code>H</code> is a placeholder — production MWIP will standardize on a different robust hash.
  </p>

  <div id="ih-upload" class="imagehide__upload">
    <input id="ih-file" type="file" accept="image/*"><br>
    <span class="imagehide__upload-sample">
      or <a href="#" id="ih-sample">load the sample image</a>
    </span>
  </div>

  <p id="ih-status" class="imagehide__status">Drop an image or pick a file to begin.</p>

  <div class="imagehide__panels">
    <div class="imagehide__panel">
      <canvas id="ih-cover"></canvas>
      <div class="imagehide__panel-label">cover</div>
    </div>
    <div class="imagehide__panel">
      <canvas id="ih-container"></canvas>
      <div class="imagehide__panel-label">container (watermarked)</div>
    </div>
    <div class="imagehide__panel">
      <canvas id="ih-residual"></canvas>
      <div class="imagehide__panel-label">residual ×10</div>
    </div>
  </div>

  <pre id="ih-oneshot" class="imagehide__oneshot">One-shot stats will appear here after encoding.</pre>

  <div class="imagehide__controls">
    <strong>Attacks:</strong>
    <span id="ih-attackList"></span>
    <button id="ih-runBtn" class="primary" disabled>Run</button>
  </div>

  <table class="imagehide__results">
    <thead>
      <tr>
        <th>Attack</th>
        <th class="num">PSNR</th>
        <th class="num">SSIM</th>
        <th class="num">Bit acc</th>
        <th>Sig</th>
      </tr>
    </thead>
    <tbody id="ih-resultsBody"></tbody>
  </table>

  <p style="margin-top:1.5rem;font-size:0.95rem;color:#555;">
    <em>Out of scope for this model:</em> crop, rotate, Gaussian noise, Gaussian blur. The MWIP encoder targets the canonical-delivery scope (JPEG + benign down-up resize).
  </p>

</section>

<script type="module" src="{{ '/assets/imagehide/app.js' | relative_url }}"></script>
```

- [ ] **Step 2: Verify all element IDs referenced by `app.js` exist in the HTML**

```bash
cd /home/hangyang_umass_edu/yanghangAI.github.io
grep -oE "id=\"ih-[a-zA-Z]+\"" _pages/imagehide.html | sort -u
grep -oE "ih-[a-zA-Z]+" assets/imagehide/app.js | sort -u
```
Compare the two lists; every ID consumed by `app.js` must appear in the HTML. Expected IDs from `app.js`:
`ih-upload, ih-file, ih-sample, ih-status, ih-cover, ih-container, ih-residual, ih-oneshot, ih-attackList, ih-runBtn, ih-resultsBody, ih-override` (the last is created dynamically — present only when over-cap).

- [ ] **Step 3: Commit**

```bash
git add _pages/imagehide.html
git commit -m "feat(imagehide): page shell with upload UI, panels, results table"
```

---

## Task 11: Catalog entries — `_tools/imagehide.md` and `_portfolio/imagehide.md`

**Working directory:** `/home/hangyang_umass_edu/yanghangAI.github.io`

**Files:**
- Create: `_tools/imagehide.md`
- Create: `_portfolio/imagehide.md`

- [ ] **Step 1: Add tool entry**

Write to `_tools/imagehide.md`:

```markdown
---
title: Imagehide demo
link: /imagehide/
date: 2026-05-16
status: live
summary: In-browser interactive demo of the MWIP INN watermark. Upload a photo, watch the model embed an 896-bit payload, then run it through 11 channel attacks (JPEG, resize, social-media chains) and see PSNR, SSIM, bit accuracy, and signature verification per attack. All client-side via ONNX Runtime Web.
---
```

- [ ] **Step 2: Add portfolio entry**

Write to `_portfolio/imagehide.md`:

```markdown
---
title: "MWIP — invertible-network image watermarking"
collection: portfolio
date: 2026-05-16
excerpt: "A 4M-param INN that embeds an 896-bit cryptographic provenance payload (perceptual hash + Ed25519 signature + public key) directly into image pixels — survives JPEG re-encoding and benign resize down to social-media chain levels."
paperurl: ""
codeurl: ""
---

Personal research project — embedded watermarking for real photos under the
canonical-delivery scope (JPEG + benign resize). Designed to survive the
Instagram / X / WhatsApp / WeChat compression chains.

[Try the live demo →](/imagehide/)
```

- [ ] **Step 3: Verify both files appear in the relevant Jekyll loops**

```bash
grep -A1 "site.tools" _layouts/home.html _pages/tools.html
grep "site.portfolio" _layouts/home.html
```
Expected: each grep returns at least one matching line; both `_tools/imagehide.md` and `_portfolio/imagehide.md` will be picked up automatically by those loops at build time.

- [ ] **Step 4: Commit**

```bash
git add _tools/imagehide.md _portfolio/imagehide.md
git commit -m "feat(imagehide): tools + portfolio catalog entries"
```

---

## Task 12: Sample cover image

**Working directory:** `/home/hangyang_umass_edu/yanghangAI.github.io`

**Files:**
- Create: `assets/imagehide/sample-cover.jpg`

Need a same-aspect-ratio photo, JPEG, ≤300 KB, with dimensions large enough that center-trim to multiples of 64 leaves a meaningful encoded region (target ~1024×768 or 1280×720). Source should be license-clean.

- [ ] **Step 1: Reuse `images/IMG_0037.jpeg` from the AmazingHand portfolio entry if it meets the criteria, otherwise pick another existing license-clean photo**

```bash
identify images/IMG_0037.jpeg 2>/dev/null || file images/IMG_0037.jpeg
ls -l images/IMG_0037.jpeg
```
If the dimensions are reasonable (≥ 768 on both axes) and the file is < 300 KB, copy it:

```bash
cp images/IMG_0037.jpeg assets/imagehide/sample-cover.jpg
```

If it's too large or not suitable, downsize:

```bash
# requires `convert` from ImageMagick or alternative
convert images/IMG_0037.jpeg -resize 1280x1280\> -quality 85 \
        assets/imagehide/sample-cover.jpg
```

If neither path is available, document the gap and proceed without a sample — the demo still works on user-uploaded images; the sample is a nicety.

- [ ] **Step 2: Verify size and dimensions**

```bash
ls -lh assets/imagehide/sample-cover.jpg
file assets/imagehide/sample-cover.jpg
```
Expected: < 300 KB JPEG, both dimensions ≥ 768.

- [ ] **Step 3: Commit**

```bash
git add assets/imagehide/sample-cover.jpg
git commit -m "feat(imagehide): sample cover image for the 'load default sample' link"
```

---

## Task 13: Push, verify Pages build, smoke-test the live page

**Working directory:** `/home/hangyang_umass_edu/yanghangAI.github.io`

- [ ] **Step 1: Inventory pending commits**

```bash
git log --oneline origin/master..HEAD
git status
```
Expected: 11+ commits ahead of origin, clean working tree.

- [ ] **Step 2: Sanity-check Liquid includes and asset paths**

```bash
# All script/css/image asset references in the new page resolve under assets/imagehide/
grep -oE "(href|src)=\"[^\"]+\"" _pages/imagehide.html | sort -u
# Ensure every file referenced exists
for p in encoder.onnx decoder.onnx app.js style.css sample-cover.jpg; do
  test -f "assets/imagehide/$p" && echo "ok: $p" || echo "MISSING: $p"
done
```
Expected: all five `ok:` lines. If `sample-cover.jpg` is missing (per Task 12 fallback), that's a soft fail — the demo still works.

- [ ] **Step 3: Push**

```bash
git push origin master
```

- [ ] **Step 4: Wait for the Pages build to complete**

```bash
until [ "$(gh run list --branch master --workflow=pages-build-deployment --limit 1 --json status -q '.[0].status')" = "completed" ]; do sleep 5; done
gh run list --branch master --workflow=pages-build-deployment --limit 1
```
Expected: `completed success`.

- [ ] **Step 5: Smoke-test the URL**

```bash
curl -sI https://yanghangAI.github.io/imagehide/ | head -1
curl -s  https://yanghangAI.github.io/imagehide/ | grep -E "<h1>|imagehide|encoder.onnx" | head -5
for p in /assets/imagehide/encoder.onnx /assets/imagehide/decoder.onnx /assets/imagehide/app.js /assets/imagehide/style.css /assets/imagehide/sample-cover.jpg; do
  printf "%-45s " "$p"
  curl -sI "https://yanghangAI.github.io$p" | head -1
done
```
Expected: page returns `HTTP/2 200`; all asset URLs except possibly `sample-cover.jpg` (per fallback in Task 12) return `HTTP/2 200`.

- [ ] **Step 6: Manual browser smoke test (user, not the agent)**

Open `https://yanghangAI.github.io/imagehide/` in Chrome (or any browser with WebGPU). Verify:
- Page renders with the upload area and the controls; status reads "Drop an image…".
- Click "load the sample image" (or upload a small JPEG).
- Status switches to "Loading model… X%" while the .onnx files fetch; progress bar advances.
- After load, status reads `Model loaded (backend: webgpu)`. Run button enables.
- Click Run. Container and residual canvases fill within a few seconds. One-shot stats line populates.
- Results table rows go from greyed-out → emboldened → final values. All 11 attacks complete.
- `identity` row has bit accuracy 1.000 and a green ✓; aggressive attacks (`chain_wechat`, `resize_050`) likely have lower bit accuracy and may show ✗.

If any of these fail, open the browser DevTools console, copy the first error, and escalate.

- [ ] **Step 7: If smoke test passes, no further commits**

If smoke test surfaces an issue:

```bash
# Make fix, commit, push
git add <files>
git commit -m "fix(imagehide): <what>"
git push origin master
```
Re-run Steps 4–6 until clean.

---

## Self-review checklist

(Recorded for the executor.)

- **Spec coverage:** Every section of the spec maps to a task.
  - Architecture diagram → Tasks 6 (attacks), 7 (pipeline), 9 (app), 10 (page)
  - Inference path / ONNX exports → Tasks 1, 2
  - Resolution handling / center-trim → Tasks 3 (trim), 9 (app integration)
  - Payload + verify policy → Tasks 4 (payload), 9 (app uses libsodium)
  - 11-attack panel → Task 6 (catalog matches spec table)
  - Metrics (attacked_container vs attacked_cover) → Tasks 5 (metrics), 9 (app loop)
  - Page integration + permalink + catalog entries → Tasks 10, 11
  - Two-stage loading UX → Task 9 (ensureLoaded + progress callbacks)
  - Aesthetic boundaries → Task 8 (scoped CSS)
  - Code layout → matches files listed in tasks
  - Risks (WebGPU, fp16 size, cold-start) → surfaced in app.js status messages (Task 9) and the WASM warning

- **Placeholder scan:** No "TBD", "TODO", "fill in later" anywhere. Every code block is complete and runnable.

- **Type consistency:**
  - `computeCrop` returns `{ cropH, cropW, top, left, trimmedTop, trimmedBottom, trimmedLeft, trimmedRight }` — same shape used by `splitTrim`, `pasteBack`, and `app.js`.
  - `packPayload(H, sig, pk)` / `unpackPayload(bits)` → `{ H, sig, pk }` — consistent in `app.js` use.
  - `encode(coverImageData, bitsUint8)` → `{ container, ms }` — consistent in `app.js` use.
  - `decode(containerImageData)` → `{ bits, ms }` — consistent in `app.js` use.
  - Attack catalog `ATTACKS` is the single source of truth for both `app.js` UI and the metrics loop.
  - ONNX I/O names (`host_rgb`, `bits`, `container_rgb`, `bit_logits`) match between Task 1 export script and Task 7 pipeline.

- **Cross-repo handoff:** Task 2 is the single coupling point between the model repo and the site repo (copies two `.onnx` files). No other task crosses the boundary.
