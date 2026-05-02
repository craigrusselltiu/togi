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

togi sprite -i                           # batch, write back over input/
togi sprite -i panda.png                 # process a single file in place
togi sprite -i some/external/file.png    # in place at an arbitrary path

togi background                          # process every image in input/
togi background saloon.png               # single file
togi background -i                       # batch in place
togi background -i saloon.png            # single file in place
```

### Commands

- `togi sprite [input_name] [output_name] [--size N] [--force] [-i|--in-place] [--palette PATH] [--input DIR] [--output DIR]`
  Runs the full sprite pipeline. `--size` defaults to 64. Output is always `.png`.
  `--palette`, `--input`, and `--output` override the corresponding values in
  `togi.toml` for this run.

- `togi background [input_name] [output_name] [--force] [-i|--in-place] [--palette PATH] [--input DIR] [--output DIR]`
  Runs the background pipeline. `--palette`, `--input`, and `--output` override
  the corresponding values in `togi.toml` for this run.

### In-place mode (`-i` / `--in-place`)

- With no `input_name`: walks `input/` recursively and overwrites each source.
- With an `input_name`: resolved as a path relative to the current directory
  if that path exists, otherwise as a name relative to `input/`. The file is
  overwritten in place.
- Sources whose extension is not `.png` are replaced by a sibling `.png` and
  the original file is removed.
- `output_name` may not be supplied with `--in-place`.

### Batch behavior

When no filename is given, walks `input/` recursively. Subdirectories are mirrored into `output/`. Supported input extensions: `.png`, `.jpg`, `.jpeg`, `.webp`. Output is always `.png` (alpha required).

### Incremental processing

By default, a file is skipped if `output/foo.png` exists and is newer than `input/foo.png`. `--force` reprocesses everything.

### Individual step subcommands (escape hatches)

For debugging or one-off use, each pipeline step is exposed as its own
subcommand. Inputs/outputs use the same handling as the full pipelines:
positional args or `--input` / `--output` flags, falling back to the dirs in
`togi.toml` when both forms are omitted. Passing both the positional and the
flag for the same role is an error.

```
togi outline                                # batch input/ -> output/
togi outline a.png                          # ./a.png -> output/a.png
togi palette in.png out.png --palette path/to/pal.hex
togi palette --input in.png --output out.png --palette path/to/pal.hex
togi fit in.png out.png --size 64
```

When the resolved input is a directory the step batches over it; when it is
a single file and the output is a directory (or defaulted from config), the
result is written under that directory using the input's basename.

## Pipeline steps

Each step is a pure function `(image[, params]) -> image`. All operate on RGBA.

### Step 1: `bg-remove` — flood fill white background

- 4-connected flood fill from all four corners.
- A pixel is "fillable" if its Euclidean RGB distance to pure white (`#ffffff`) is `< 30`.
- Filled pixels become `(0, 0, 0, 0)`.
- Interior whites (eye highlights, teeth, etc.) are preserved because they are not edge-connected.

### Step 2: `cleanup` — halo and speck removal

Three passes on the alpha mask:

1. **Halo pass**: for each pixel adjacent to a transparent pixel, re-check the white-distance with a tighter threshold (`< 15`). If it matches, make it transparent. Kills the faint anti-aliased ring left after step 1.
2. **Interior-pocket pass**: label connected components of opaque near-white pixels (`< 15` of pure white). Drop any component larger than `200` pixels — these are negative-space pockets enclosed by the silhouette (gaps between legs, under arms, inside cloaks). Smaller components survive so eye highlights, teeth, and gun glints are preserved.
3. **Speck pass**: connected-components label on the alpha mask (alpha > 0). Drop any component smaller than 4 pixels.

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

### `strip-watermark` — chop off a fixed bottom-right region

Crops the bottom `40px` and rightmost `100px` from the image. Used by the
background pipeline to strip the Gemini watermark before resize. Sized to
cover the typical Gemini sparkle badge with margin to spare.

### `resize` — cover-fit to 680×380

1. Scale the image so that *both* axes reach or exceed the target
   (`max(target_w / w, target_h / h)`), using nearest-neighbor (preserves
   crisp pixel edges, no anti-aliasing before palette snap).
2. Center-crop the overflow on the longer axis.

Always produces a `680 × 380` RGBA image. Used by the background pipeline.

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
strip-watermark -> resize -> palette
```

Output is always `680 × 380`. No transparency: backgrounds are assumed to
fill the whole canvas, so `bg-remove` and `cleanup` are not run.

## Hardcoded constants

- White flood-fill tolerance: `30` (RGB Euclidean)
- Halo cleanup tolerance: `15`
- Min connected component size: `4` pixels
- Max interior near-white pocket size before drop: `200` pixels
- Alpha binarization threshold: `128`
- Default sprite size: `64`
- Outline color: `#000000`
- Outline thickness: `1px`
- Background output size: `680 × 380`
- Watermark strip: `100px` from right, `40px` from bottom

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
