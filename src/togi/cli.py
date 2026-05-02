from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from . import config as config_mod
from . import imageio, palette, pipeline, steps
from .errors import TogiError
from .steps import DEFAULT_SIZE


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="togi",
        description="Sharpen Gemini-generated game sprites and backgrounds.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sprite", help="run the full sprite pipeline")
    sp.add_argument("input_name", nargs="?", default=None)
    sp.add_argument("output_name", nargs="?", default=None)
    sp.add_argument("--size", type=int, default=DEFAULT_SIZE)
    sp.add_argument("--force", action="store_true")
    sp.add_argument(
        "-i",
        "--in-place",
        action="store_true",
        help="overwrite source files instead of writing to output/",
    )
    sp.add_argument(
        "--palette",
        default=None,
        dest="palette_path",
        help="palette .hex file; overrides the palette in togi.toml",
    )
    sp.add_argument(
        "--input",
        default=None,
        dest="input_dir",
        help="directory to read inputs from; overrides input in togi.toml",
    )
    sp.add_argument(
        "--output",
        default=None,
        dest="output_dir",
        help="directory to write outputs to; overrides output in togi.toml",
    )
    sp.add_argument(
        "--chroma-weight",
        type=float,
        default=1.0,
        help="chroma-loss penalty for palette snap; 0 disables (default 1.0)",
    )

    bg = sub.add_parser("background", help="run the background pipeline")
    bg.add_argument("input_name", nargs="?", default=None)
    bg.add_argument("output_name", nargs="?", default=None)
    bg.add_argument("--force", action="store_true")
    bg.add_argument(
        "-i",
        "--in-place",
        action="store_true",
        help="overwrite source files instead of writing to output/",
    )
    bg.add_argument(
        "--palette",
        default=None,
        dest="palette_path",
        help="palette .hex file; overrides the palette in togi.toml",
    )
    bg.add_argument(
        "--input",
        default=None,
        dest="input_dir",
        help="directory to read inputs from; overrides input in togi.toml",
    )
    bg.add_argument(
        "--output",
        default=None,
        dest="output_dir",
        help="directory to write outputs to; overrides output in togi.toml",
    )
    bg.add_argument(
        "--chroma-weight",
        type=float,
        default=1.0,
        help="chroma-loss penalty for palette snap; 0 disables (default 1.0)",
    )

    def add_step_io(s: argparse.ArgumentParser) -> None:
        s.add_argument("input_pos", nargs="?", default=None, metavar="input")
        s.add_argument("output_pos", nargs="?", default=None, metavar="output")
        s.add_argument(
            "--input",
            default=None,
            dest="input_flag",
            help="input path (file or directory); falls back to togi.toml",
        )
        s.add_argument(
            "--output",
            default=None,
            dest="output_flag",
            help="output path (file or directory); falls back to togi.toml",
        )
        s.add_argument("--force", action="store_true")
        s.add_argument(
            "-i",
            "--in-place",
            action="store_true",
            help="overwrite input in place; input may be a file or directory",
        )

    for name in (
        "bg-remove",
        "cleanup",
        "crop-bbox",
        "outline",
        "strip-watermark",
        "resize",
    ):
        s = sub.add_parser(name, help=f"run only the {name} step")
        add_step_io(s)

    fit = sub.add_parser("fit", help="run only the fit step")
    add_step_io(fit)
    fit.add_argument("--size", type=int, default=DEFAULT_SIZE)

    pal = sub.add_parser("palette", help="run only the palette-snap step")
    add_step_io(pal)
    pal.add_argument(
        "--palette",
        default=None,
        dest="palette_path",
        help="palette .hex file; defaults to the palette in togi.toml",
    )
    pal.add_argument(
        "--chroma-weight",
        type=float,
        default=1.0,
        help="chroma-loss penalty for palette snap; 0 disables (default 1.0)",
    )

    return p


def _dispatch_pipeline(
    cfg, args: argparse.Namespace, run
) -> int:
    if args.in_place:
        if args.output_name is not None:
            raise TogiError(
                "--in-place does not accept an output_name argument"
            )
        if args.input_name is None:
            return pipeline.run_batch_in_place(cfg.input, run)
        return pipeline.run_single_in_place(cfg.input, run, args.input_name)
    if args.input_name is None:
        return pipeline.run_batch(cfg.input, cfg.output, run, force=args.force)
    return pipeline.run_single(
        cfg.input,
        cfg.output,
        run,
        args.input_name,
        args.output_name,
        force=args.force,
    )


def _override_dirs(cfg, args: argparse.Namespace):
    overrides = {}
    if args.input_dir is not None:
        overrides["input"] = Path(args.input_dir).resolve()
    if args.output_dir is not None:
        overrides["output"] = Path(args.output_dir).resolve()
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


def _cmd_sprite(args: argparse.Namespace) -> int:
    cfg = _override_dirs(config_mod.load(), args)
    palette_rgb = palette.parse_hex(args.palette_path or cfg.palette)

    def run(img):
        return pipeline.sprite_pipeline(
            img,
            size=args.size,
            palette_rgb=palette_rgb,
            chroma_weight=args.chroma_weight,
        )

    return _dispatch_pipeline(cfg, args, run)


def _cmd_background(args: argparse.Namespace) -> int:
    cfg = _override_dirs(config_mod.load(), args)
    palette_rgb = palette.parse_hex(args.palette_path or cfg.palette)

    def run(img):
        return pipeline.background_pipeline(
            img,
            palette_rgb=palette_rgb,
            chroma_weight=args.chroma_weight,
        )

    return _dispatch_pipeline(cfg, args, run)


def _build_step_runner(args: argparse.Namespace):
    if args.cmd == "bg-remove":
        return steps.bg_remove
    if args.cmd == "cleanup":
        return steps.cleanup
    if args.cmd == "crop-bbox":
        return steps.crop_bbox
    if args.cmd == "outline":
        return steps.outline
    if args.cmd == "strip-watermark":
        return steps.strip_watermark
    if args.cmd == "resize":
        return steps.resize
    if args.cmd == "fit":
        size = args.size
        return lambda img: steps.fit(img, size=size)
    if args.cmd == "palette":
        path = args.palette_path or config_mod.load().palette
        palette_rgb = palette.parse_hex(path)
        chroma_weight = args.chroma_weight
        return lambda img: steps.palette_snap(
            img, palette_rgb=palette_rgb, chroma_weight=chroma_weight
        )
    raise AssertionError(f"unknown step: {args.cmd}")


def _resolve_step_io(args: argparse.Namespace) -> tuple[str | None, str | None]:
    if args.input_pos is not None and args.input_flag is not None:
        raise TogiError("input given twice (positional and --input)")
    if args.output_pos is not None and args.output_flag is not None:
        raise TogiError("output given twice (positional and --output)")
    return (
        args.input_flag or args.input_pos,
        args.output_flag or args.output_pos,
    )


def _cmd_step(args: argparse.Namespace) -> int:
    run = _build_step_runner(args)
    input_raw, output_raw = _resolve_step_io(args)

    cfg = None

    def get_cfg():
        nonlocal cfg
        if cfg is None:
            cfg = config_mod.load()
        return cfg

    in_path = Path(input_raw) if input_raw else get_cfg().input

    if args.in_place:
        if output_raw is not None:
            raise TogiError(
                "--in-place does not accept an output argument"
            )
        return pipeline.run_path_in_place(str(in_path), run)

    out_path = Path(output_raw) if output_raw else get_cfg().output

    if not in_path.exists():
        raise TogiError(f"input not found: {in_path}")

    if in_path.is_dir():
        if out_path.exists() and not out_path.is_dir():
            raise TogiError(
                f"output must be a directory when input is a directory: {out_path}"
            )
        return pipeline.run_batch(in_path, out_path, run, force=args.force)

    if output_raw is None or out_path.is_dir():
        out_path = out_path / in_path.with_suffix(".png").name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = imageio.load(in_path)
    result = run(img)
    imageio.save(result, out_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    print("Processing...", flush=True)
    try:
        if args.cmd == "sprite":
            rc = _cmd_sprite(args)
        elif args.cmd == "background":
            rc = _cmd_background(args)
        else:
            rc = _cmd_step(args)
    except TogiError as e:
        print(f"togi: {e}", file=sys.stderr)
        return 1
    if rc == 0:
        print("Completed.", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
