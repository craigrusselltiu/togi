from __future__ import annotations

import pytest

from togi import config as config_mod
from togi.errors import TogiError


def test_load_missing_file_errors(tmp_path):
    with pytest.raises(TogiError, match="togi.toml not found"):
        config_mod.load(cwd=tmp_path)


def test_load_missing_key_errors(tmp_path):
    (tmp_path / "togi.toml").write_text(
        'input = "raw/"\noutput = "sprites/"\n'
    )
    with pytest.raises(TogiError, match="palette"):
        config_mod.load(cwd=tmp_path)


def test_load_resolves_relative_paths(tmp_path):
    (tmp_path / "togi.toml").write_text(
        'input = "raw/"\noutput = "sprites/"\npalette = "p.hex"\n'
    )
    cfg = config_mod.load(cwd=tmp_path)
    assert cfg.input == tmp_path / "raw"
    assert cfg.output == tmp_path / "sprites"
    assert cfg.palette == tmp_path / "p.hex"
