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
