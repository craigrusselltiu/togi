from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import TogiError


def load(path: str | Path) -> np.ndarray:
    path = str(path)
    src = sys.stdin.buffer if path == "-" else path
    try:
        with Image.open(src) as img:
            return np.array(img.convert("RGBA"))
    except FileNotFoundError:
        raise TogiError(f"input file not found: {path}")


def save(arr: np.ndarray, path: str | Path) -> None:
    path = str(path)
    img = Image.fromarray(arr, "RGBA")
    if path == "-":
        img.save(sys.stdout.buffer, format="PNG")
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img.save(path, format="PNG")
