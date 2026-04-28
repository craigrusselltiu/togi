from __future__ import annotations

import numpy as np
import pytest

from togi import steps
from togi.errors import FullyTransparentError


def test_bg_remove_clears_white_border_keeps_interior(make):
    img = make(8, 8, fill=(255, 255, 255, 255))
    img[3:5, 3:5] = (200, 50, 50, 255)
    out = steps.bg_remove(img)
    assert out[0, 0, 3] == 0
    assert out[7, 7, 3] == 0
    assert tuple(out[3, 3]) == (200, 50, 50, 255)


def test_bg_remove_preserves_interior_white_island(make):
    img = make(8, 8, fill=(255, 255, 255, 255))
    img[2:6, 2:6] = (10, 10, 10, 255)
    img[3:5, 3:5] = (255, 255, 255, 255)
    out = steps.bg_remove(img)
    assert out[0, 0, 3] == 0
    assert tuple(out[2, 2]) == (10, 10, 10, 255)
    assert tuple(out[3, 3]) == (255, 255, 255, 255)


def test_bg_remove_handles_no_white_corner(make):
    img = make(4, 4, fill=(50, 50, 50, 255))
    out = steps.bg_remove(img)
    np.testing.assert_array_equal(out, img)


def test_cleanup_drops_3px_speck_keeps_4px(make):
    img = make(10, 10)
    img[1, 1] = (100, 100, 100, 255)
    img[1, 2] = (100, 100, 100, 255)
    img[2, 1] = (100, 100, 100, 255)
    img[6:8, 6:8] = (100, 100, 100, 255)
    out = steps.cleanup(img)
    assert out[1, 1, 3] == 0
    assert out[1, 2, 3] == 0
    assert out[2, 1, 3] == 0
    assert out[6, 6, 3] == 255
    assert out[7, 7, 3] == 255


def test_cleanup_halo_pass_kills_near_white_adjacent_to_transparent(make):
    img = make(6, 6)
    img[2:4, 2:4] = (200, 50, 50, 255)
    img[1, 2] = (250, 250, 250, 255)
    img[1, 3] = (250, 250, 250, 255)
    img[2, 1] = (250, 250, 250, 255)
    img[3, 1] = (250, 250, 250, 255)
    out = steps.cleanup(img)
    assert out[1, 2, 3] == 0
    assert out[1, 3, 3] == 0
    assert out[2, 1, 3] == 0
    assert out[3, 1, 3] == 0
    assert tuple(out[2, 2]) == (200, 50, 50, 255)


def test_crop_bbox_square_input_yields_square_output(make):
    img = make(10, 10)
    img[3:6, 4:8] = (100, 100, 100, 255)
    out = steps.crop_bbox(img)
    assert out.shape[0] == out.shape[1], "square input must yield square crop"
    assert out.shape[0] >= 4 and out.shape[1] >= 4


def test_crop_bbox_preserves_input_aspect_ratio(make):
    img = make(20, 40)
    img[5:9, 10:14] = (100, 100, 100, 255)
    out = steps.crop_bbox(img)
    h, w, _ = out.shape
    assert w / h == pytest.approx(40 / 20, rel=0.1)


def test_crop_bbox_pads_with_transparent_at_edge(make):
    # Tall-thin subject hugging the left edge: the square-up expansion would
    # need to extend past column 0, so the result must include transparent
    # padding on the left.
    img = make(20, 20)
    img[2:18, 0:4] = (100, 100, 100, 255)
    out = steps.crop_bbox(img)
    assert out.shape[0] == out.shape[1]
    assert (out[..., 3] == 0).any(), "edge crop must include transparent padding"
    assert (out[..., 3] > 0).any()


def test_crop_bbox_raises_on_fully_transparent(make):
    img = make(4, 4)
    with pytest.raises(FullyTransparentError):
        steps.crop_bbox(img)


def test_fit_outputs_size_with_binary_alpha_and_transparent_border(make):
    img = make(10, 10, fill=(200, 50, 50, 255))
    out = steps.fit(img, size=16)
    assert out.shape == (16, 16, 4)
    assert (out[0, :, 3] == 0).all()
    assert (out[-1, :, 3] == 0).all()
    assert (out[:, 0, 3] == 0).all()
    assert (out[:, -1, 3] == 0).all()
    unique_alphas = set(np.unique(out[..., 3]).tolist())
    assert unique_alphas <= {0, 255}


def test_fit_upscale_uses_nearest(make):
    img = make(4, 4, fill=(200, 50, 50, 255))
    out = steps.fit(img, size=16)
    assert out.shape == (16, 16, 4)
    inner = out[1:-1, 1:-1]
    rgb_in_inner = inner[inner[..., 3] > 0][..., :3]
    np.testing.assert_array_equal(
        rgb_in_inner, np.full_like(rgb_in_inner, fill_value=0) + np.array([200, 50, 50])
    )


def test_outline_draws_3x3_ring_around_single_opaque_pixel(make):
    img = make(5, 5)
    img[2, 2] = (200, 50, 50, 255)
    out = steps.outline(img)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                assert tuple(out[2, 2]) == (200, 50, 50, 255)
            else:
                assert tuple(out[2 + dy, 2 + dx]) == (0, 0, 0, 255)


def test_strip_watermark_crops_bottom_right(make):
    img = make(200, 400, fill=(50, 50, 50, 255))
    out = steps.strip_watermark(img)
    assert out.shape == (
        200 - steps.WATERMARK_BOTTOM,
        400 - steps.WATERMARK_RIGHT,
        4,
    )


def test_resize_cover_fits_to_target_dims(make):
    img = make(792, 1408, fill=(40, 60, 120, 255))
    out = steps.resize(img)
    assert out.shape == (steps.BG_TARGET_H, steps.BG_TARGET_W, 4)


def test_resize_upscales_too_small_input(make):
    img = make(100, 200, fill=(10, 20, 30, 255))
    out = steps.resize(img)
    assert out.shape == (steps.BG_TARGET_H, steps.BG_TARGET_W, 4)


def test_palette_snap_snaps_to_nearest_and_zeroes_transparent(make):
    img = make(4, 4)
    img[0, 0] = (210, 30, 30, 255)
    img[1, 1] = (50, 50, 50, 100)
    palette_rgb = np.array(
        [[0, 0, 0], [255, 255, 255], [200, 50, 50], [50, 100, 200]],
        dtype=np.uint8,
    )
    out = steps.palette_snap(img, palette_rgb=palette_rgb)
    assert tuple(out[0, 0]) == (200, 50, 50, 255)
    assert tuple(out[1, 1]) == (0, 0, 0, 0)
