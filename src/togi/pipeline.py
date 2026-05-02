from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

from . import imageio, steps
from .errors import TogiError

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def sprite_pipeline(
    img: np.ndarray,
    *,
    size: int,
    palette_rgb: np.ndarray,
    chroma_weight: float = 1.0,
) -> np.ndarray:
    img = steps.bg_remove(img)
    img = steps.cleanup(img)
    img = steps.crop_bbox(img)
    img = steps.palette_snap(
        img, palette_rgb=palette_rgb, chroma_weight=chroma_weight
    )
    img = steps.fit(img, size=size)
    img = steps.outline(img)
    return img


def background_pipeline(
    img: np.ndarray,
    *,
    palette_rgb: np.ndarray,
    chroma_weight: float = 1.0,
) -> np.ndarray:
    img = steps.strip_watermark(img)
    img = steps.resize(img)
    img = steps.palette_snap(
        img, palette_rgb=palette_rgb, chroma_weight=chroma_weight
    )
    return img


def _is_up_to_date(src: Path, dst: Path) -> bool:
    return dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime


def _process_one(
    src: Path,
    dst: Path,
    pipeline: Callable[[np.ndarray], np.ndarray],
) -> None:
    img = imageio.load(src)
    out = pipeline(img)
    imageio.save(out, dst)


def run_single(
    input_dir: Path,
    output_dir: Path,
    pipeline: Callable[[np.ndarray], np.ndarray],
    input_name: str,
    output_name: str | None,
    *,
    force: bool,
) -> int:
    src = input_dir / input_name
    if not src.exists():
        raise TogiError(f"input file not found: {src}")
    rel = Path(output_name) if output_name else Path(input_name)
    dst = output_dir / rel.with_suffix(".png")
    if not force and _is_up_to_date(src, dst):
        return 0
    try:
        _process_one(src, dst, pipeline)
    except TogiError as e:
        print(f"{src}: {e}", file=sys.stderr)
        return 1
    return 0


def run_batch(
    input_dir: Path,
    output_dir: Path,
    pipeline: Callable[[np.ndarray], np.ndarray],
    *,
    force: bool,
) -> int:
    if not input_dir.exists():
        raise TogiError(f"input directory not found: {input_dir}")

    failures = 0
    found_any = False
    for src in sorted(input_dir.rglob("*")):
        if not src.is_file():
            continue
        if src.suffix.lower() not in SUPPORTED_EXTS:
            continue
        found_any = True
        rel = src.relative_to(input_dir)
        dst = output_dir / rel.with_suffix(".png")
        if not force and _is_up_to_date(src, dst):
            continue
        try:
            _process_one(src, dst, pipeline)
        except TogiError as e:
            print(f"{src}: {e}", file=sys.stderr)
            failures += 1

    if not found_any:
        print(f"no input images found in {input_dir}", file=sys.stderr)
    return 1 if failures else 0


def _process_in_place(
    src: Path,
    pipeline: Callable[[np.ndarray], np.ndarray],
) -> None:
    dst = src.with_suffix(".png")
    _process_one(src, dst, pipeline)
    if src != dst and src.exists():
        src.unlink()


def run_single_in_place(
    input_dir: Path,
    pipeline: Callable[[np.ndarray], np.ndarray],
    name_or_path: str,
) -> int:
    given = Path(name_or_path)
    if given.exists():
        src = given
    else:
        src = input_dir / name_or_path
        if not src.exists():
            raise TogiError(f"input file not found: {given}")
    try:
        _process_in_place(src, pipeline)
    except TogiError as e:
        print(f"{src}: {e}", file=sys.stderr)
        return 1
    return 0


def run_batch_in_place(
    input_dir: Path,
    pipeline: Callable[[np.ndarray], np.ndarray],
) -> int:
    return _walk_in_place(input_dir, pipeline)


def _walk_in_place(
    directory: Path,
    pipeline: Callable[[np.ndarray], np.ndarray],
) -> int:
    if not directory.exists():
        raise TogiError(f"input directory not found: {directory}")

    failures = 0
    found_any = False
    for src in sorted(directory.rglob("*")):
        if not src.is_file():
            continue
        if src.suffix.lower() not in SUPPORTED_EXTS:
            continue
        found_any = True
        try:
            _process_in_place(src, pipeline)
        except TogiError as e:
            print(f"{src}: {e}", file=sys.stderr)
            failures += 1

    if not found_any:
        print(f"no input images found in {directory}", file=sys.stderr)
    return 1 if failures else 0


def run_path_in_place(
    path: str,
    pipeline: Callable[[np.ndarray], np.ndarray],
) -> int:
    p = Path(path)
    if not p.exists():
        raise TogiError(f"path not found: {p}")
    if p.is_dir():
        return _walk_in_place(p, pipeline)
    try:
        _process_in_place(p, pipeline)
    except TogiError as e:
        print(f"{p}: {e}", file=sys.stderr)
        return 1
    return 0
