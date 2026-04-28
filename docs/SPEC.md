# togi

研ぎ — *sharpening, polishing*.

A CLI tool for processing game sprites and backgrounds generated from Gemini (or any source on a white background) into clean, palette-snapped game assets.

## Goals

Take a square image with a white background and produce a transparent, cropped, outlined, palette-snapped sprite ready to drop into a game. Steps are modular so non-sprite assets (backgrounds) can use a subset of the pipeline.

## Tech stack

- **Language**: Python 3.11+
- **Dependencies**: Pillow, NumPy, SciPy (connected components, morphology), `tomllib` (stdlib)
- **Distribution**: local CLI, run via `uv` or a venv

## Configuration

A single `togi.toml` file in the current working directory holds **only**:

```toml
input = "raw/"
output = "sprites/"
palette = "palettes/omitc.hex"
```

- `input`: directory containing source images
- `output`: directory where processed images are written
- `palette`: path to a `.hex` palette file

Nothing else goes in the config. Mode, size, tolerances, etc. are CLI flags or hardcoded defaults.

## Palette format

Plain text `.hex`, one `#RRGGBB` per line. Blank lines and lines starting with `;` or `#` (when used as a comment, not a color) are ignored.

```
; OMITC core palette
#000000
#ffffff
#c83e3e
#3e6ec8
```

**Validation**: `#000000` must be present in the palette. Error out otherwise — the outline step depends on it.

## CLI

Filenames passed on the CLI are relative to the configured `input`/`output` dirs.

```
togi sprite                              # process every image in input/
togi sprite panda.png                    # raw/panda.png -> sprites/panda.png
togi sprite panda.png panda_v2.png       # raw/panda.png -> sprites/panda_v2.png
togi sprite --size 64                    # set sprite size (default 64)
togi sprite --force                      # reprocess even if output is newer

togi background                          # process every image in input/
togi background saloon.png               # single file
```

### Commands

- `togi sprite [input_name] [output_name] [--size N] [--force]`
  Runs the full sprite pipeline. `--size` defaults to 64. Output is always `.png`.

- `togi background [input_name] [output_name] [--force]`
  Runs the background pipeline (no crop, no resize, no outline).

### Batch behavior

When no filename is given, walks `input/` recursively. Subdirectories are mirrored into `output/`. Supported input extensions: `.png`, `.jpg`, `.jpeg`, `.webp`. Output is always `.png` (alpha required).

### Incremental processing

By default, a file is skipped if `output/foo.png` exists and is newer than `input/foo.png`. `--force` reprocesses everything.

### Individual step subcommands (escape hatches)

For debugging or one-off use, each pipeline step is exposed as its own subcommand taking explicit input/output paths (no config lookup):

```
togi bg-remove in.png out.png
togi cleanup in.png out.png
togi crop-bbox in.png out.png
togi fit in.png out.png --size 64
togi outline in.png out.png
togi palette in.png out.png --palette path/to/pal.hex
```

These can be piped:

```
togi bg-remove in.png - | togi cleanup - - | togi palette - out.png --palette omitc.hex
```

## Pipeline steps

Each step is a pure function `(image[, params]) -> image`. All operate on RGBA.

### Step 1: `bg-remove` — flood fill white background

- 4-connected flood fill from all four corners.
- A pixel is "fillable" if its Euclidean RGB distance to pure white (`#ffffff`) is `< 30`.
- Filled pixels become `(0, 0, 0, 0)`.
- Interior whites (eye highlights, teeth, etc.) are preserved because they are not edge-connected.

### Step 2: `cleanup` — halo and speck removal

Two passes on the alpha mask:

1. **Halo pass**: for each pixel adjacent to a transparent pixel, re-check the white-distance with a tighter threshold (`< 15`). If it matches, make it transparent. Kills the faint anti-aliased ring left after step 1.
2. **Speck pass**: connected-components label on the alpha mask (alpha > 0). Drop any component smaller than 4 pixels.

### Step 3: `crop-bbox` — crop to non-transparent bounding box

- Compute the tightest rectangle containing all pixels with `alpha > 0`.
- Expand that rectangle to match the **input image's aspect ratio**, centered on the tight bbox center. A square input always yields a square crop, preserving subject proportions through `fit`.
- If the expanded rectangle extends past the image edge, fill the missing region with fully transparent pixels.
- If the image is fully transparent, error out with a clear message.

### Step 4: `fit` — resize and pad to target size

Given a target size `N` (e.g. 64):

1. Resize the bbox-cropped image to `(N - 2, N - 2)`.
   - **Downscale**: Lanczos.
   - **Upscale**: nearest neighbor (preserves crispness).
2. Re-threshold alpha: `alpha > 128` → `255`, else → `0`. Gives a hard edge for the outline to latch onto.
3. Add 1px of fully transparent padding on all four sides, producing an `N × N` image.

The 1px padding ring guarantees the outline (step 5) never eats into sprite pixels.

### Step 5: `outline` — 1px black outline

- Build a binary mask from alpha (`> 0`).
- Dilate the mask by 1px using a 3×3 kernel.
- Pixels in the dilated mask but not in the original become `(0, 0, 0, 255)`.

### Step 6: `palette` — snap colors to palette

- For each pixel:
  - If `alpha < 128`: set to `(0, 0, 0, 0)` (fully transparent).
  - If `alpha >= 128`: set alpha to `255`, then snap RGB to the nearest palette color.
- Distance metric: **OkLab**. Convert each pixel and each palette entry once, take Euclidean distance in OkLab.
- No dithering.

The alpha-binarization here gives sprites guaranteed clean edges in the final output.

## Pipeline composition

### Sprite pipeline (`togi sprite`)

```
bg-remove -> cleanup -> crop-bbox -> fit -> outline -> palette
```

### Background pipeline (`togi background`)

```
bg-remove -> cleanup -> palette
```

No cropping, no resize, no outline. Preserves original canvas dimensions.

## Hardcoded constants

- White flood-fill tolerance: `30` (RGB Euclidean)
- Halo cleanup tolerance: `15`
- Min connected component size: `4` pixels
- Alpha binarization threshold: `128`
- Default sprite size: `64`
- Outline color: `#000000`
- Outline thickness: `1px`

These are not exposed as flags. If they need to change, edit the source.

## Error cases

- Missing `togi.toml` in cwd → error with example config.
- Missing `input`/`output`/`palette` keys → error naming the missing key.
- Palette file missing or unparseable → error with line number.
- `#000000` not in palette → error.
- Fully transparent image after `bg-remove` + `cleanup` → error naming the file.
- Input file not found (when named on CLI) → error with the resolved path.

## Out of scope (for v1)

- Configurable tolerances or thresholds in the config file.
- Per-file overrides.
- Dithering.
- Non-square sprite output.
- Animated sprite sheets.
- Watch mode.
