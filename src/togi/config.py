from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .errors import TogiError

CONFIG_NAME = "togi.toml"

EXAMPLE = """\
input = "raw/"
output = "sprites/"
palette = "palettes/omitc.hex"
"""


@dataclass(frozen=True)
class Config:
    input: Path
    output: Path
    palette: Path


def load(cwd: Path | None = None) -> Config:
    base = Path(cwd) if cwd is not None else Path.cwd()
    path = base / CONFIG_NAME
    if not path.exists():
        raise TogiError(
            f"{CONFIG_NAME} not found in {base}\n\n"
            f"create one with:\n\n{EXAMPLE}"
        )
    with path.open("rb") as f:
        data = tomllib.load(f)

    for key in ("input", "output", "palette"):
        if key not in data:
            raise TogiError(f"{CONFIG_NAME} missing required key: {key!r}")

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (base / pp)

    return Config(
        input=_resolve(data["input"]),
        output=_resolve(data["output"]),
        palette=_resolve(data["palette"]),
    )
