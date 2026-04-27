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


def test_crop_bbox_tightens_to_opaque_region(make):
    img = make(10, 10)
    img[3:6, 4:8] = (100, 100, 100, 255)
    out = steps.crop_bbox(img)
    assert out.shape == (3, 4, 4)
    assert (out[..., 3] == 255).all()


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
