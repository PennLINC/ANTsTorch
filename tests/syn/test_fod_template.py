"""Tests for FOD (spherical harmonic) template construction with build_fod_template."""
import os

import ants
import numpy as np
import pytest
import torch

from antstorch.syn import build_fod_template, sh

_DIRECTION = np.diag([-1.0, -1.0, 1.0])
_CENTERS = ((6, 6, 7), (5, 7, 8), (7, 6, 6))
# A plain gradient-descent affine (no multi-start): the multi-start search can pick a 180-degree rotated
# optimum on these nearly symmetric toy shapes, and ANTs' affine averaging then decomposes it into negative
# scales.
FAST = dict(levels=(2, 1), reg_iterations=(3, 2), affine_iterations=(10, 10, 10), affine_multi_start=False)


def _blob(shape, center, sigma):
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float64)
    return np.exp(-(((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) / (2.0 * sigma**2)))


def _fod_image(shape, center, lmax=2, direction_index=3):
    """A blob plus a smaller satellite (asymmetric, so the affine cannot land on a reflection), one fiber."""
    directions = sh.load_directions(60, dtype=torch.float64)
    fiber = sh.apsf_matrix(directions, lmax)[:, direction_index].numpy()
    satellite = tuple(c + o for c, o in zip(center, (2.5, -2.0, 1.5)))
    weights = _blob(shape, center, 3.0) + 0.6 * _blob(shape, satellite, 1.5)
    array = weights[..., None] * fiber[None, None, None, :]
    array[..., 0] += 0.05 * weights
    array = np.ascontiguousarray(np.transpose(array, (2, 1, 0, 3)).astype(np.float32))
    return ants.from_numpy(
        array, origin=(-20.0, -18.0, 5.0), spacing=(2.0, 2.0, 2.0), direction=_DIRECTION, has_components=True
    )


def _scalar_image(shape, center, sigma=2.5):
    array = np.ascontiguousarray(np.transpose(_blob(shape, center, sigma), (2, 1, 0)).astype(np.float32))
    return ants.from_numpy(array, origin=(-20.0, -18.0, 5.0), spacing=(2.0, 2.0, 2.0), direction=_DIRECTION)


def test_build_fod_template_returns_a_vector_template(tmp_path):
    images = [_fod_image((12, 13, 14), c) for c in _CENTERS]
    result = build_fod_template(
        image_list=images, iterations=2, gradient_step=0.2, blending_weight=0.75, output_dir=str(tmp_path), **FAST
    )
    template = result["template"]
    assert template.components == images[0].components and tuple(template.shape) == tuple(images[0].shape)
    assert np.isfinite(template.numpy()).all()
    assert result["n_iterations"] == 2 and len(result["convergence"]) == 2
    assert all(w.components == images[0].components for w in result["warped_images"])
    assert os.path.exists(os.path.join(str(tmp_path), "avgWarp_0.nii.gz"))


def test_build_fod_template_rejects_mismatched_components(tmp_path):
    images = [_fod_image((8, 8, 8), (4, 4, 4), lmax=4), _fod_image((8, 8, 8), (4, 4, 4), lmax=2)]
    with pytest.raises(ValueError, match="same number"):
        build_fod_template(image_list=images, iterations=1, output_dir=str(tmp_path), levels=(1,), reg_iterations=(1,))


@pytest.mark.slow
def test_build_fod_template_with_an_extra_contrast_and_pre_alignment(tmp_path):
    images = [_fod_image((12, 13, 14), c) for c in _CENTERS]
    extras = [_scalar_image((12, 13, 14), c) * 3.0 for c in _CENTERS]
    result = build_fod_template(
        image_list=images, extra_contrasts=[extras], iterations=1, blending_weight=0.75, initial_alignment="Rigid",
        output_dir=str(tmp_path), **FAST,
    )
    assert result["template"].components == images[0].components
    assert result["initial_alignment"]["template"].components == images[0].components
    extra_template = result["extra_templates"][0]
    assert extra_template.components == 1 and tuple(extra_template.shape) == tuple(images[0].shape)
    assert np.isfinite(result["template"].numpy()).all() and np.isfinite(extra_template.numpy()).all()
