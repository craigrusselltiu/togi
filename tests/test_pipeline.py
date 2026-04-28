from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from togi import pipeline
from togi.config import Config
from togi.errors import TogiError


def _identity(img: np.ndarray) -> np.ndarray:
    return img


def _save_png(path: Path, fill=(200, 50, 50, 255)) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[..., :] = fill
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGBA").save(path, format="PNG")


def _save_jpg(path: Path, fill=(200, 50, 50)) -> None:
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[..., :] = fill
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGB").save(path, format="JPEG")


def _cfg(tmp_path: Path) -> Config:
    return Config(
        input=tmp_path / "raw",
        output=tmp_path / "out",
        palette=tmp_path / "p.hex",
    )


def test_run_batch_writes_to_output_with_same_filenames(tmp_path):
    cfg = _cfg(tmp_path)
    _save_png(cfg.input / "a.png")
    _save_png(cfg.input / "sub" / "b.png")

    rc = pipeline.run_batch(cfg, _identity, force=False)
    assert rc == 0
    assert (cfg.output / "a.png").exists()
    assert (cfg.output / "sub" / "b.png").exists()
    assert (cfg.input / "a.png").exists()


def test_run_batch_in_place_overwrites_sources(tmp_path):
    cfg = _cfg(tmp_path)
    _save_png(cfg.input / "a.png")
    _save_png(cfg.input / "sub" / "b.png")

    rc = pipeline.run_batch_in_place(cfg, _identity)
    assert rc == 0
    assert (cfg.input / "a.png").exists()
    assert (cfg.input / "sub" / "b.png").exists()
    assert not cfg.output.exists()


def test_run_batch_in_place_converts_jpg_to_png_and_removes_original(tmp_path):
    cfg = _cfg(tmp_path)
    _save_jpg(cfg.input / "a.jpg")

    rc = pipeline.run_batch_in_place(cfg, _identity)
    assert rc == 0
    assert (cfg.input / "a.png").exists()
    assert not (cfg.input / "a.jpg").exists()


def test_run_single_in_place_resolves_filename_against_input(tmp_path):
    cfg = _cfg(tmp_path)
    _save_png(cfg.input / "panda.png")

    rc = pipeline.run_single_in_place(cfg, _identity, "panda.png")
    assert rc == 0
    assert (cfg.input / "panda.png").exists()
    assert not cfg.output.exists()


def test_run_single_in_place_uses_explicit_path_outside_input(tmp_path):
    cfg = _cfg(tmp_path)
    target = tmp_path / "elsewhere" / "x.png"
    _save_png(target)

    rc = pipeline.run_single_in_place(cfg, _identity, str(target))
    assert rc == 0
    assert target.exists()


def test_run_single_in_place_jpg_becomes_png(tmp_path):
    cfg = _cfg(tmp_path)
    target = tmp_path / "lone.jpg"
    _save_jpg(target)

    rc = pipeline.run_single_in_place(cfg, _identity, str(target))
    assert rc == 0
    assert (tmp_path / "lone.png").exists()
    assert not target.exists()


def test_run_single_in_place_missing_file_errors(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.input.mkdir()
    with pytest.raises(TogiError, match="input file not found"):
        pipeline.run_single_in_place(cfg, _identity, "nope.png")


def test_run_batch_in_place_missing_input_dir_errors(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(TogiError, match="input directory not found"):
        pipeline.run_batch_in_place(cfg, _identity)
