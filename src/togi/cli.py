from __future__ import annotations

import argparse
import sys

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

    def add_step_io(s: argparse.ArgumentParser) -> None:
        s.add_argument("input")
        s.add_argument("output", nargs="?", default=None)
        s.add_argument(
            "-i",
            "--in-place",
            action="store_true",
            help="overwrite input in place; input may be a file or directory",
        )

    for name in ("bg-remove", "cleanup", "crop-bbox", "outline"):
        s = sub.add_parser(name, help=f"run only the {name} step")
        add_step_io(s)

    fit = sub.add_parser("fit", help="run only the fit step")
    add_step_io(fit)
    fit.add_argument("--size", type=int, default=DEFAULT_SIZE)

    pal = sub.add_parser("palette", help="run only the palette-snap step")
    add_step_io(pal)
    pal.add_argument("--palette", required=True, dest="palette_path")

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
            return pipeline.run_batch_in_place(cfg, run)
        return pipeline.run_single_in_place(cfg, run, args.input_name)
    if args.input_name is None:
        return pipeline.run_batch(cfg, run, force=args.force)
    return pipeline.run_single(
        cfg, run, args.input_name, args.output_name, force=args.force
    )


def _cmd_sprite(args: argparse.Namespace) -> int:
    cfg = config_mod.load()
    palette_rgb = palette.parse_hex(cfg.palette)

    def run(img):
        return pipeline.sprite_pipeline(
            img, size=args.size, palette_rgb=palette_rgb
        )

    return _dispatch_pipeline(cfg, args, run)


def _cmd_background(args: argparse.Namespace) -> int:
    cfg = config_mod.load()
    palette_rgb = palette.parse_hex(cfg.palette)

    def run(img):
        return pipeline.background_pipeline(img, palette_rgb=palette_rgb)

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
    if args.cmd == "fit":
        size = args.size
        return lambda img: steps.fit(img, size=size)
    if args.cmd == "palette":
        palette_rgb = palette.parse_hex(args.palette_path)
        return lambda img: steps.palette_snap(img, palette_rgb=palette_rgb)
    raise AssertionError(f"unknown step: {args.cmd}")


def _cmd_step(args: argparse.Namespace) -> int:
    run = _build_step_runner(args)

    if args.in_place:
        if args.output is not None:
            raise TogiError(
                "--in-place does not accept an output argument"
            )
        return pipeline.run_path_in_place(args.input, run)

    if args.output is None:
        raise TogiError(
            f"{args.cmd}: output path required (or pass --in-place)"
        )

    img = imageio.load(args.input)
    out = run(img)
    imageio.save(out, args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "sprite":
            return _cmd_sprite(args)
        if args.cmd == "background":
            return _cmd_background(args)
        return _cmd_step(args)
    except TogiError as e:
        print(f"togi: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
