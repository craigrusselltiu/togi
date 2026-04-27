from __future__ import annotations

import numpy as np
import pytest


def make_rgba(h: int, w: int, fill=(0, 0, 0, 0)) -> np.ndarray:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :] = fill
    return arr


@pytest.fixture
def make():
    return make_rgba
