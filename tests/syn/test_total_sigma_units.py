"""``total_sigma`` shares ``flow_sigma``'s units, selected by ``gaussian_sigma_mode`` (voxels of the current level by default)."""
import numpy as np
import pytest

import ants
from antstorch.syn import syn_registration


def _blob_image(center, spacing, shape=(14, 14, 14), sigma=2.5):
    grid = np.indices(shape, dtype=np.float32)
    array = np.exp(-sum((g - c) ** 2 for g, c in zip(grid, center)) / (2.0 * sigma ** 2))
    return ants.from_numpy(np.ascontiguousarray(array), spacing=spacing)


@pytest.mark.parametrize("spacing", [2.0, 0.5])
def test_total_sigma_in_voxel_mode_equals_physical_mode_scaled_by_spacing(spacing):
    fixed = _blob_image((7, 7, 7), (spacing,) * 3)
    moving = _blob_image((6, 7, 8), (spacing,) * 3)
    kw = dict(type_of_transform="SyNOnly", syn_metric="mse", levels=(1,), reg_iterations=(6,), grad_step=0.2, device="cpu")
    voxel = syn_registration(fixed, moving, flow_sigma=1.5, total_sigma=1.0, gaussian_sigma_mode="voxel", **kw)
    physical = syn_registration(fixed, moving, flow_sigma=1.5 * spacing, total_sigma=1.0 * spacing, gaussian_sigma_mode="physical", **kw)
    stiffer = syn_registration(fixed, moving, flow_sigma=1.5, total_sigma=2.0, gaussian_sigma_mode="voxel", **kw)
    warp_voxel = ants.image_read(voxel["fwdtransforms"][0]).numpy()
    assert np.abs(warp_voxel).max() > 0.05 * spacing
    np.testing.assert_allclose(warp_voxel, ants.image_read(physical["fwdtransforms"][0]).numpy(), atol=1e-5)
    assert not np.allclose(warp_voxel, ants.image_read(stiffer["fwdtransforms"][0]).numpy(), atol=1e-3)
