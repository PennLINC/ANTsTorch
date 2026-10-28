"""The internal affine fit uses the extra contrasts as further channels (``affine_extra_contrasts``)."""
import numpy as np

import ants
from antstorch.syn import syn_registration


def _blob_image(center, shape=(16, 16, 16), sigma=2.5):
    grid = np.indices(shape, dtype=np.float32)
    array = np.exp(-sum((g - c) ** 2 for g, c in zip(grid, center)) / (2.0 * sigma ** 2))
    return ants.from_numpy(np.ascontiguousarray(array), spacing=(1.0, 1.0, 1.0))


def _translation_fit(affine_extra_contrasts):
    # The primary images agree (a broad centered blob), only the extra contrast is shifted: an affine fit
    # that reads the extra contrast moves, one that reads the primary alone stays put.
    primary = _blob_image((8, 8, 8), sigma=4.0)
    extra_fixed = _blob_image((8, 8, 8), sigma=1.5)
    extra_moving = _blob_image((8, 8, 11), sigma=1.5)
    result = syn_registration(
        primary, primary, type_of_transform="Translation", extra_contrasts=[(extra_fixed, extra_moving)],
        affine_extra_contrasts=affine_extra_contrasts, affine_multi_start=False, device="cpu",
    )
    return result["affine_translation"].numpy()


def test_extra_contrasts_drive_the_affine_fit_by_default():
    assert np.abs(_translation_fit(True)).max() > 1.0


def test_extra_contrasts_can_be_kept_out_of_the_affine_fit():
    assert np.abs(_translation_fit(False)).max() < 0.2


def test_syn_with_extra_contrasts_still_runs_through_the_joint_affine():
    primary_fixed = _blob_image((8, 8, 8))
    primary_moving = _blob_image((8, 8, 9))
    extra_fixed = _blob_image((8, 8, 8), sigma=1.5)
    extra_moving = _blob_image((8, 8, 9), sigma=1.5)
    result = syn_registration(
        primary_fixed, primary_moving, type_of_transform="SyN", extra_contrasts=[(extra_fixed, extra_moving)],
        syn_metric="mse", levels=(1,), reg_iterations=(2,), affine_multi_start=False, device="cpu",
    )
    assert result["affine_matrix"].shape == (3, 3)
    assert len(result["warpedmovout_extras"]) == 1
    assert np.isfinite(result["warpedmovout_extras"][0].numpy()).all()
