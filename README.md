# togi

研ぎ — *sharpening, polishing*.

A CLI tool that turns Gemini-generated (or any white-background) images into
clean, transparent, palette-snapped game sprites and backgrounds.

See [`docs/SPEC.md`](docs/SPEC.md) for the full specification.

## Install

```sh
uv venv
uv pip install -e .
```

Or with pip in any Python 3.11+ environment:

```sh
pip install -e .
```

For development (adds pytest):

```sh
uv pip install -e ".[dev]"
pytest
```

## Configuration

Create `togi.toml` in your working directory:

```toml
input = "raw/"
output = "sprites/"
palette = "palettes/omitc.hex"
```

The palette file is plain text, one `#RRGGBB` per line. Lines starting with
`;` or `#` (when not a valid color) are comments. `#000000` must be present
because the outline step depends on it.

## Usage

```sh
togi sprite                          # process every image in input/
togi sprite panda.png                # raw/panda.png -> sprites/panda.png
togi sprite panda.png panda_v2.png   # rename on output
togi sprite --size 64                # set sprite size (default 64)
togi sprite --force                  # reprocess even if output is newer

togi sprite -i                       # process every image in input/ in place
togi sprite -i panda.png             # process raw/panda.png in place
togi sprite -i path/to/file.png      # process the given path in place

togi background                      # background pipeline (no crop/resize/outline)
togi background saloon.png
togi background -i                   # batch in place
togi background -i saloon.png        # single in place
```

Filenames passed on the CLI are relative to `input/` and `output/`. Output is
always `.png`. Input extensions: `.png`, `.jpg`, `.jpeg`, `.webp`. Subdirectories
are mirrored.

By default, files are skipped when `output/foo.png` is newer than the input.
`--force` reprocesses everything.

With `-i`/`--in-place`, files are written back over their source location and
no `output_name` may be supplied. Non-`.png` sources (e.g. `.jpg`) are replaced
by a sibling `.png` (the original is removed). When `-i` is given a single
argument, it is resolved as a path relative to the current directory if that
path exists, otherwise as a name relative to `input/`.

### Per-step subcommands

Each pipeline step is exposed as its own subcommand for debugging. They take
explicit paths (no config lookup) and support `-` for stdin/stdout piping:

```sh
togi bg-remove in.png out.png
togi cleanup in.png out.png
togi crop-bbox in.png out.png
togi fit in.png out.png --size 64
togi outline in.png out.png
togi palette in.png out.png --palette palettes/omitc.hex

togi bg-remove in.png - | togi cleanup - - | togi palette - out.png --palette omitc.hex
```

Per-step subcommands also accept `-i`/`--in-place`. The argument may be a
single file or a directory; with a directory, every supported image inside
is processed in place (recursively).

```sh
togi palette -i ./scratch --palette palettes/omitc.hex   # every image in ./scratch
togi outline -i ./scratch/foo.png                        # single file in place
togi fit -i ./scratch --size 64
```

## Pipelines

- **sprite**: `bg-remove → cleanup → crop-bbox → fit → outline → palette`
- **background**: `bg-remove → cleanup → palette`
