from __future__ import annotations

import numpy as np
import pytest

from togi import palette
from togi.errors import TogiError


def test_parse_hex_basic(tmp_path):
    p = tmp_path / "p.hex"
    p.write_text(
        "; comment\n"
        "# also a comment\n"
        "\n"
        "#000000\n"
        "#ffffff\n"
        "#c83e3e\n"
    )
    arr = palette.parse_hex(p)
    assert arr.shape == (3, 3)
    assert tuple(arr[0]) == (0, 0, 0)
    assert tuple(arr[1]) == (255, 255, 255)
    assert tuple(arr[2]) == (200, 62, 62)


def test_parse_hex_missing_black_errors(tmp_path):
    p = tmp_path / "p.hex"
    p.write_text("#ffffff\n#c83e3e\n")
    with pytest.raises(TogiError, match="#000000"):
        palette.parse_hex(p)


def test_parse_hex_bad_line_errors_with_lineno(tmp_path):
    p = tmp_path / "p.hex"
    p.write_text("#000000\nnot-a-color\n")
    with pytest.raises(TogiError, match=":2:"):
        palette.parse_hex(p)


def test_parse_hex_missing_file(tmp_path):
    with pytest.raises(TogiError, match="not found"):
        palette.parse_hex(tmp_path / "nope.hex")


def test_rgb_to_oklab_returns_float32_3channel():
    rgb = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0]], dtype=np.uint8)
    lab = palette.rgb_to_oklab(rgb)
    assert lab.shape == (3, 3)
    assert lab.dtype == np.float32
    # White has L close to 1.0
    assert lab[1, 0] == pytest.approx(1.0, abs=0.01)
    # Black has L close to 0.0
    assert lab[0, 0] == pytest.approx(0.0, abs=0.01)
