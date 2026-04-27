from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from .errors import FullyTransparentError

WHITE_TOL = 30
HALO_TOL = 15
MIN_COMPONENT = 4
ALPHA_THRESHOLD = 128
DEFAULT_SIZE = 64

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
    return rgba[r0 : r1 + 1, c0 : c1 + 1].copy()


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


def outline(rgba: np.ndarray) -> np.ndarray:
    opaque = rgba[..., 3] > 0
    dilated = ndimage.binary_dilation(opaque, structure=SQUARE3)
    ring = dilated & ~opaque
    out = rgba.copy()
    out[ring] = (0, 0, 0, 255)
    return out


def palette_snap(rgba: np.ndarray, *, palette_rgb: np.ndarray) -> np.ndarray:
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
    dists = np.linalg.norm(
        pixels_lab[:, None, :] - palette_lab[None, :, :], axis=2
    )
    nearest = palette_rgb[np.argmin(dists, axis=1)]

    snapped = out[opaque]
    snapped[..., :3] = nearest
    out[opaque] = snapped
    return out
