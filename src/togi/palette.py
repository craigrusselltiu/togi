from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .errors import TogiError

HEX_RE = re.compile(r"^#([0-9a-fA-F]{6})$")

M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float32,
)

M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float32,
)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_oklab(rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert (..., 3) sRGB uint8 to (..., 3) float32 OkLab."""
    rgb = rgb_uint8.astype(np.float32) / 255.0
    lin = srgb_to_linear(rgb).astype(np.float32)
    lms = lin @ M1.T
    lms_cbrt = np.cbrt(lms).astype(np.float32)
    return (lms_cbrt @ M2.T).astype(np.float32)


def parse_hex(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise TogiError(f"palette file not found: {path}")

    colors: list[tuple[int, int, int]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(";"):
            continue
        m = HEX_RE.match(line)
        if m:
            hexstr = m.group(1)
            colors.append(
                (int(hexstr[0:2], 16), int(hexstr[2:4], 16), int(hexstr[4:6], 16))
            )
            continue
        if line.startswith("#"):
            # `#`-prefixed lines that aren't valid `#RRGGBB` are comments.
            continue
        raise TogiError(f"{path}:{lineno}: cannot parse palette line: {raw!r}")

    if not colors:
        raise TogiError(f"{path}: palette is empty")

    arr = np.array(colors, dtype=np.uint8)
    if not np.any(np.all(arr == 0, axis=1)):
        raise TogiError(
            f"{path}: palette must include #000000 (required by outline step)"
        )
    return arr
