from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from togi.cli import main


def _save_png(path: Path, fill=(210, 30, 30, 255)) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[..., :] = fill
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGBA").save(path, format="PNG")


@pytest.fixture
def in_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_config(root: Path, palette_name: str = "pal.hex") -> None:
    (root / "togi.toml").write_text(
        f'input = "scratch/"\noutput = "out/"\npalette = "{palette_name}"\n'
    )
    (root / palette_name).write_text("#000000\n#c83e3e\n")


def test_palette_step_uses_config_palette_when_flag_omitted(in_cwd):
    _write_config(in_cwd)
    _save_png(in_cwd / "scratch" / "a.png")

    rc = main(["palette", "-i", "./scratch"])
    assert rc == 0
    assert (in_cwd / "scratch" / "a.png").exists()
    out = np.array(Image.open(in_cwd / "scratch" / "a.png").convert("RGBA"))
    # Should snap to palette red (#c83e3e == 200, 62, 62)
    assert tuple(out[0, 0, :3]) == (200, 62, 62)


def test_palette_step_explicit_flag_overrides_config(in_cwd):
    _write_config(in_cwd)
    (in_cwd / "other.hex").write_text("#000000\n#3e6ec8\n")
    _save_png(in_cwd / "scratch" / "a.png", fill=(50, 100, 200, 255))

    rc = main(
        ["palette", "-i", "./scratch", "--palette", "other.hex"]
    )
    assert rc == 0
    out = np.array(Image.open(in_cwd / "scratch" / "a.png").convert("RGBA"))
    assert tuple(out[0, 0, :3]) == (62, 110, 200)


def test_step_in_place_rejects_output_arg(in_cwd, capsys):
    _save_png(in_cwd / "a.png")
    rc = main(["outline", "-i", "a.png", "out.png"])
    assert rc == 1
    assert "--in-place" in capsys.readouterr().err


def test_step_without_output_falls_back_to_config_output(in_cwd):
    _write_config(in_cwd)
    _save_png(in_cwd / "a.png")
    rc = main(["outline", "a.png"])
    assert rc == 0
    assert (in_cwd / "out" / "a.png").exists()


def test_step_without_input_or_output_uses_config_dirs(in_cwd):
    _write_config(in_cwd)
    _save_png(in_cwd / "scratch" / "a.png")
    _save_png(in_cwd / "scratch" / "sub" / "b.png")
    rc = main(["outline"])
    assert rc == 0
    assert (in_cwd / "out" / "a.png").exists()
    assert (in_cwd / "out" / "sub" / "b.png").exists()


def test_step_without_input_uses_config_input_in_place(in_cwd):
    _write_config(in_cwd)
    _save_png(in_cwd / "scratch" / "a.png")
    rc = main(["outline", "-i"])
    assert rc == 0
    assert (in_cwd / "scratch" / "a.png").exists()


def test_step_accepts_input_output_flags(in_cwd):
    _save_png(in_cwd / "a.png")
    rc = main(["outline", "--input", "a.png", "--output", "b.png"])
    assert rc == 0
    assert (in_cwd / "b.png").exists()


def test_palette_step_accepts_input_flag(in_cwd):
    _write_config(in_cwd)
    _save_png(in_cwd / "a.png")
    rc = main(
        ["palette", "--input", "a.png", "--output", "b.png", "--palette", "pal.hex"]
    )
    assert rc == 0
    out = np.array(Image.open(in_cwd / "b.png").convert("RGBA"))
    assert tuple(out[0, 0, :3]) == (200, 62, 62)


def test_step_input_given_twice_errors(in_cwd, capsys):
    _save_png(in_cwd / "a.png")
    rc = main(["outline", "a.png", "--input", "a.png", "--output", "b.png"])
    assert rc == 1
    assert "input given twice" in capsys.readouterr().err


def test_step_in_place_with_input_flag(in_cwd):
    _save_png(in_cwd / "a.png")
    rc = main(["outline", "-i", "--input", "a.png"])
    assert rc == 0
    assert (in_cwd / "a.png").exists()
