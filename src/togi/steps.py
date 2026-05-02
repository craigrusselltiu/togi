from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from .errors import FullyTransparentError

WHITE_TOL = 30
HALO_TOL = 15
MIN_COMPONENT = 4
INTERIOR_WHITE_MAX = 200
ALPHA_THRESHOLD = 128
DEFAULT_SIZE = 64
BG_TARGET_W = 680
BG_TARGET_H = 380
WATERMARK_RIGHT = 100
WATERMARK_BOTTOM = 40

CROSS4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
SQUARE3 = np.ones((3, 3), dtype=bool)


def _white_distance(rgb: np.ndarray) -> np.ndarray:
    diff = 255.0 - rgb.astype(np.float32)
    return np.sqrt((diff * diff).sum(axis=-1))


def bg_remove(rgba: np.ndarray) -> np.ndarray:
    mask = _white_distance(rgba[..., :3]) < WHITE_TOL
    labels, _ = ndimage.label(mask, structure=CROSS4)
    h, w = labels.shape
    corner_labels = {
        int(labels[0, 0]),
        int(labels[0, w - 1]),
        int(labels[h - 1, 0]),
        int(labels[h - 1, w - 1]),
    }
    corner_labels.discard(0)
    out = rgba.copy()
    if corner_labels:
        fill = np.isin(labels, list(corner_labels))
        out[fill] = 0
    return out


def cleanup(rgba: np.ndarray) -> np.ndarray:
    out = rgba.copy()

    transparent = out[..., 3] == 0
    if transparent.any():
        adj = ndimage.binary_dilation(transparent, structure=CROSS4) & ~transparent
        candidates = adj & (_white_distance(out[..., :3]) < HALO_TOL)
        out[candidates] = 0

    # Interior near-white pockets (limb gaps, clothing folds) survived bg-remove
    # because they aren't corner-connected. Drop the large ones; keep small ones
    # so eye highlights, teeth, and glints survive.
    interior_white = (out[..., 3] > 0) & (_white_distance(out[..., :3]) < HALO_TOL)
    if interior_white.any():
        labels, n = ndimage.label(interior_white, structure=CROSS4)
        if n > 0:
            sizes = np.bincount(labels.ravel())
            big = np.isin(labels, np.where(sizes > INTERIOR_WHITE_MAX)[0])
            out[big & interior_white] = 0

    opaque = out[..., 3] > 0
    labels, n = ndimage.label(opaque, structure=CROSS4)
    if n > 0:
        sizes = np.bincount(labels.ravel())
        small = np.isin(labels, np.where(sizes < MIN_COMPONENT)[0]) & opaque
        out[small] = 0

    return out


def crop_bbox(rgba: np.ndarray) -> np.ndarray:
    opaque = rgba[..., 3] > 0
    if not opaque.any():
        raise FullyTransparentError("image is fully transparent")
    rows = np.any(opaque, axis=1)
    cols = np.any(opaque, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    img_h, img_w = rgba.shape[:2]
    bbox_h = int(r1 - r0 + 1)
    bbox_w = int(c1 - c0 + 1)

    # Expand the tight bbox to the input image's aspect ratio so a square
    # input always yields a square crop (no horizontal/vertical stretch in fit).
    if bbox_w * img_h >= bbox_h * img_w:
        new_w = bbox_w
        new_h = -(-bbox_w * img_h // img_w)  # ceil
    else:
        new_h = bbox_h
        new_w = -(-bbox_h * img_w // img_h)

    extra_h = new_h - bbox_h
    extra_w = new_w - bbox_w
    top = extra_h // 2
    left = extra_w // 2
    nr0 = int(r0) - top
    nc0 = int(c0) - left
    nr1 = nr0 + new_h
    nc1 = nc0 + new_w

    out = np.zeros((new_h, new_w, 4), dtype=rgba.dtype)
    sr0, sr1 = max(nr0, 0), min(nr1, img_h)
    sc0, sc1 = max(nc0, 0), min(nc1, img_w)
    if sr1 > sr0 and sc1 > sc0:
        out[sr0 - nr0 : sr1 - nr0, sc0 - nc0 : sc1 - nc0] = rgba[sr0:sr1, sc0:sc1]
    return out


def fit(rgba: np.ndarray, *, size: int = DEFAULT_SIZE) -> np.ndarray:
    target = size - 2
    h, w = rgba.shape[:2]
    img = Image.fromarray(rgba, "RGBA")
    # Lanczos when shrinking, Nearest when growing. Spec is ambiguous on mixed
    # axes; using max() picks Lanczos whenever any axis would be downscaled.
    resample = Image.LANCZOS if max(w, h) > target else Image.NEAREST
    img = img.resize((target, target), resample=resample)
    arr = np.array(img)
    arr[..., 3] = np.where(arr[..., 3] > ALPHA_THRESHOLD, 255, 0).astype(np.uint8)
    return np.pad(arr, ((1, 1), (1, 1), (0, 0)), constant_values=0)


def strip_watermark(rgba: np.ndarray) -> np.ndarray:
    h, w = rgba.shape[:2]
    new_h = max(1, h - WATERMARK_BOTTOM)
    new_w = max(1, w - WATERMARK_RIGHT)
    return rgba[:new_h, :new_w].copy()


def resize(
    rgba: np.ndarray,
    *,
    target_w: int = BG_TARGET_W,
    target_h: int = BG_TARGET_H,
) -> np.ndarray:
    h, w = rgba.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w = max(target_w, round(w * scale))
    new_h = max(target_h, round(h * scale))
    img = Image.fromarray(rgba, "RGBA").resize((new_w, new_h), Image.NEAREST)
    arr = np.array(img)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return arr[top : top + target_h, left : left + target_w].copy()


def outline(rgba: np.ndarray) -> np.ndarray:
    opaque = rgba[..., 3] > 0
    dilated = ndimage.binary_dilation(opaque, structure=SQUARE3)
    ring = dilated & ~opaque
    out = rgba.copy()
    out[ring] = (0, 0, 0, 255)
    return out


def palette_snap(
    rgba: np.ndarray,
    *,
    palette_rgb: np.ndarray,
    chroma_weight: float = 1.0,
) -> np.ndarray:
    from .palette import rgb_to_oklab

    out = rgba.copy()
    alpha = out[..., 3]
    opaque = alpha >= ALPHA_THRESHOLD
    out[~opaque] = 0
    if not opaque.any():
        return out
    out[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)

    pixels = out[opaque][..., :3]
    pixels_lab = rgb_to_oklab(pixels)
    palette_lab = rgb_to_oklab(palette_rgb)
    c_pal = (
        np.sqrt(palette_lab[:, 1] ** 2 + palette_lab[:, 2] ** 2)
        if chroma_weight > 0
        else None
    )

    # Chunk to cap memory: with a 2048x2048 source and a ~70-color palette the
    # full diff tensor would exceed 40GB. 100k pixels per chunk keeps it under
    # ~85MB regardless of source size or palette length.
    nearest_idx = np.empty(len(pixels), dtype=np.int32)
    CHUNK = 100_000
    for i in range(0, len(pixels), CHUNK):
        j = min(i + CHUNK, len(pixels))
        diff = pixels_lab[i:j, None, :] - palette_lab[None, :, :]
        sq = (diff * diff).sum(axis=2)
        if chroma_weight > 0:
            c_src = np.sqrt(
                pixels_lab[i:j, 1] ** 2 + pixels_lab[i:j, 2] ** 2
            )
            loss = np.maximum(0.0, c_src[:, None] - c_pal[None, :])
            sq = sq + chroma_weight * (loss * loss)
        nearest_idx[i:j] = np.argmin(sq, axis=1)
    nearest = palette_rgb[nearest_idx]

    snapped = out[opaque]
    snapped[..., :3] = nearest
    out[opaque] = snapped
    return out
