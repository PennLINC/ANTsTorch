"""Tests for antstorch.syn.fod: FOD (spherical harmonic) I/O, reorientation and fod_registration."""
import math
import warnings

import ants
import numpy as np
import pytest
import torch

from antstorch.syn import fod_registration, sh
from antstorch.syn.bridge import ants_image_metadata, ants_image_to_tensor
from antstorch.syn.core.grid import get_physical_grid_torch
from antstorch.syn.fod import (
    FODReorientation,
    default_fod_lmax,
    direction_transform,
    fod_image_read,
    fod_image_write,
    pullback_jacobian,
    sh_lmax_of_image,
)

_DIRECTION = np.diag([-1.0, -1.0, 1.0])  # what ITK reports for a plain RAS NIfTI
_ORIGIN = (-20.0, -18.0, 5.0)
FAST = dict(type_of_transform="SyNOnly", levels=(2, 1), reg_iterations=(6, 4))


def _blob(shape, center, sigma):
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float64)
    return np.exp(-(((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) / (2.0 * sigma**2)))


def _fod_image(shape, center, lmax=4, direction_index=3, spacing=(2.0, 2.0, 2.0)):
    """A Gaussian blob carrying one aPSF fiber everywhere (channel-last (x, y, z, n) ANTs vector image)."""
    directions = sh.load_directions(60, dtype=torch.float64)
    fiber = sh.apsf_matrix(directions, lmax)[:, direction_index].numpy()
    weights = _blob(shape, center, 3.0)
    array = weights[..., None] * fiber[None, None, None, :]
    array[..., 0] += 0.05 * weights
    array = np.ascontiguousarray(np.transpose(array, (2, 1, 0, 3)).astype(np.float32))
    return ants.from_numpy(array, origin=_ORIGIN, spacing=spacing, direction=_DIRECTION, has_components=True)


def _scalar_image(shape, center, sigma=2.5):
    """A scalar Gaussian blob on the same frame as :func:`_fod_image`."""
    array = np.ascontiguousarray(np.transpose(_blob(shape, center, sigma), (2, 1, 0)).astype(np.float32))
    return ants.from_numpy(array, origin=_ORIGIN, spacing=(2.0, 2.0, 2.0), direction=_DIRECTION)


def _peak(coefficients, lmax, probe):
    return probe[int(torch.argmax(sh.sh_basis(probe, lmax) @ coefficients))]


# --- I/O and degree schedule ---------------------------------------------------


def test_fod_image_read_write_roundtrip_keeps_coefficients_and_geometry(tmp_path):
    image = _fod_image((6, 7, 8), (3, 3, 4))
    path = str(tmp_path / "fod.nii.gz")
    fod_image_write(image, path)
    back = fod_image_read(path)
    assert back.components == image.components and tuple(back.shape) == tuple(image.shape)
    np.testing.assert_allclose(back.numpy(), image.numpy(), atol=1e-6)
    np.testing.assert_allclose(back.origin, image.origin, atol=1e-5)
    np.testing.assert_allclose(back.spacing, image.spacing, atol=1e-6)
    np.testing.assert_allclose(np.asarray(back.direction), np.asarray(image.direction), atol=1e-6)
    assert sh_lmax_of_image(back) == 4


@pytest.mark.parametrize("extension", [".odx", ".mif.gz"])
def test_fod_image_read_write_roundtrip_through_odx(tmp_path, extension):
    pytest.importorskip("odx")
    image = _fod_image((6, 7, 8), (3, 3, 4))
    path = str(tmp_path / f"fod{extension}")
    fod_image_write(image, path)
    back = fod_image_read(path)
    # odx stores voxels in RAS+ order, so compare through the geometry rather than array to array
    resampled = ants.resample_image_to_target(ants.split_channels(back)[1], ants.split_channels(image)[1])
    np.testing.assert_allclose(resampled.numpy(), image.numpy()[..., 1], atol=1e-5)
    assert back.components == image.components and sh_lmax_of_image(back) == 4
    np.testing.assert_allclose(np.abs(np.asarray(back.direction)), np.eye(3), atol=1e-6)
    np.testing.assert_allclose(back.spacing, image.spacing, atol=1e-6)
    with pytest.raises(ValueError, match="mif.gz"):
        fod_image_write(image, str(tmp_path / "fod.mif"))


def test_fod_image_read_warns_about_fixels_in_an_odx_dataset(tmp_path):
    odx = pytest.importorskip("odx")
    image = _fod_image((6, 7, 8), (3, 3, 4))
    array = np.asarray(image.numpy(), dtype=np.float32)
    mask = np.ascontiguousarray(np.any(array != 0, axis=-1), dtype=np.uint8)
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.block(
        [[np.asarray(image.direction) * np.asarray(image.spacing)[None, :], np.asarray(image.origin)[:, None]],
         [np.zeros((1, 3)), np.ones((1, 1))]]
    )
    dataset = odx.from_sh_coefficients(
        array, mask=mask, affine=affine, basis="tournier07", sh_order=4, legacy=False, compute_peaks=True
    )
    assert dataset.nb_peaks > 0
    path = str(tmp_path / "with_fixels.odx")
    odx.save(path, dataset)
    with pytest.warns(UserWarning, match="fixels"):
        back = fod_image_read(path)
    assert back.components == image.components
    # the files antstorch writes carry no fixels, so reading them back is silent
    fod_image_write(back, str(tmp_path / "plain.odx"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fod_image_read(str(tmp_path / "plain.odx"))


def test_default_fod_lmax_follows_mrregister():
    assert default_fod_lmax(3, 8) == (0, 2, 4)
    assert default_fod_lmax(4, 8) == (0, 2, 4, 4)
    assert default_fod_lmax(3, 2) == (0, 2, 2)
    assert default_fod_lmax(2, 8, max_lmax=8) == (0, 2)


def test_fod_reorientation_level_lmax_and_channels():
    assert [FODReorientation(8).level_lmax(i) for i in range(4)] == [0, 2, 4, 4]
    assert [FODReorientation(8, lmax=8).level_lmax(i) for i in range(5)] == [0, 2, 4, 6, 8]
    explicit = FODReorientation(4, lmax=(2, 4))
    assert [explicit.level_lmax(i) for i in range(2)] == [2, 4]
    with pytest.raises(ValueError, match="more levels"):
        explicit.level_lmax(2)
    with pytest.raises(ValueError, match="even"):
        FODReorientation(4, lmax=(0, 6))
    image = torch.arange(15.0).view(1, 15, 1, 1, 1)  # lmax 4
    for level_index, n in ((0, 1), (1, 6), (2, 15)):
        assert torch.equal(FODReorientation(4).channels(image, level_index), image[:, :n])
    assert repr(FODReorientation(8)) == "FODReorientation(image_lmax=8, lmax=4, directions=60, modulate=False)"


# --- Jacobians and frames ------------------------------------------------------


def test_pullback_jacobian_recovers_an_affine_field_with_rotated_direction():
    # u(x) = (A - I) x + b in physical LPS coordinates: the Jacobian of x + u must be A everywhere inside.
    torch.manual_seed(0)
    angle = math.radians(20.0)
    direction = np.array(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    ) @ np.diag([-1.0, 1.0, 1.0])
    spacing = (1.5, 2.0, 2.5)
    shape_zyx = (7, 8, 9)
    X = get_physical_grid_torch(shape_zyx, spacing, (3.0, -4.0, 5.0), direction, dtype=torch.float64)
    A_xyz = torch.eye(3, dtype=torch.float64) + 0.1 * torch.randn(3, 3, dtype=torch.float64)
    A_zyx = A_xyz.flip(0).flip(1)
    u = X @ (A_zyx - torch.eye(3, dtype=torch.float64)).T + torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    jac = pullback_jacobian(u, spacing, direction)
    assert jac.shape == (1,) + shape_zyx + (3, 3)
    inner = jac[0, 1:-1, 1:-1, 1:-1]
    assert torch.allclose(inner, A_xyz.expand_as(inner), atol=1e-9)
    # an extra affine (reversed component order) multiplies from the left
    M_zyx = torch.eye(3, dtype=torch.float64) + 0.05 * torch.randn(3, 3, dtype=torch.float64)
    jac2 = pullback_jacobian(u, spacing, direction, affine=M_zyx)
    assert torch.allclose(jac2[0, 2, 3, 4], M_zyx.flip(0).flip(1) @ A_xyz, atol=1e-9)


def test_direction_transform_is_the_conjugated_inverse():
    torch.manual_seed(1)
    jac = torch.eye(3, dtype=torch.float64) + 0.2 * torch.randn(2, 3, 3, dtype=torch.float64)
    flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))
    assert torch.allclose(direction_transform(jac, "LPS"), torch.linalg.inv(jac), atol=1e-9)
    assert torch.allclose(direction_transform(jac, "RAS"), flip @ torch.linalg.inv(jac) @ flip, atol=1e-9)
    singular = torch.zeros(1, 3, 3, dtype=torch.float64)
    assert torch.isfinite(direction_transform(singular)).all()
    with pytest.raises(ValueError):
        direction_transform(jac, "XYZ")


def test_midpoint_hook_rotates_fibers_like_the_warp_and_passes_trailing_channels():
    """A pure rotation of the half-warp must rotate the FOD peaks by the same rotation (in RAS)."""
    lmax = 4
    image = _fod_image((12, 12, 12), (6, 6, 6), lmax=lmax)
    meta = ants_image_metadata(image)
    X = get_physical_grid_torch(
        meta["torch_shape"], meta["spacing"], meta["origin"], meta["direction"], dtype=torch.float64
    )
    angle = math.radians(40.0)
    R_lps = torch.tensor(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    u = X @ (R_lps.flip(0).flip(1) - torch.eye(3, dtype=torch.float64)).T  # pull-back phi(x) = R x
    I = ants_image_to_tensor(image, dtype=torch.float64, normalize=False)
    extra = torch.rand(1, 1, *I.shape[2:], dtype=torch.float64)  # an extra contrast riding along
    reorientation = FODReorientation(lmax, lmax=(lmax,))
    hook = reorientation.midpoint_hook(0, meta, torch.eye(3, dtype=torch.float64))
    out, _ = hook(torch.cat([I, extra], dim=1), torch.cat([I, extra], dim=1), u, u)
    assert torch.equal(out[:, -1:], extra)
    probe = sh.load_directions(300, dtype=torch.float64)
    peak_in = _peak(I[0, :, 6, 6, 6], lmax, probe)
    peak_out = _peak(out[0, :-1, 6, 6, 6], lmax, probe)
    # directions live in RAS; the LPS rotation R maps to flip R flip, and the fiber moves by its inverse
    flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))
    expected = flip @ torch.linalg.inv(R_lps) @ flip @ peak_in
    assert float(torch.dot(peak_out, expected).abs()) > math.cos(math.radians(7.0))  # 300-direction probe
    assert FODReorientation(lmax, lmax=(0,)).midpoint_hook(0, meta, torch.eye(3, dtype=torch.float64)) is None


def test_reorient_warped_applies_the_affine():
    """Pulling an FOD image back through a rotation about z must rotate every peak (in RAS) by its inverse."""
    lmax = 4
    image = _fod_image((12, 12, 12), (6, 6, 6), lmax=lmax)
    meta = ants_image_metadata(image)
    angle = math.radians(30.0)
    A = torch.tensor(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    tensor = ants_image_to_tensor(image, dtype=torch.float64, normalize=False)
    displacement = torch.zeros(1, *meta["torch_shape"], 3, dtype=torch.float64)
    out = FODReorientation(lmax).reorient_warped(tensor, displacement, meta, affine=A.flip(0).flip(1))
    probe = sh.load_directions(300, dtype=torch.float64)
    flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))
    expected = flip @ torch.linalg.inv(A) @ flip @ _peak(tensor[0, :, 6, 6, 6], lmax, probe)
    peak_after = _peak(out[0, :, 6, 6, 6], lmax, probe)
    assert float(torch.dot(peak_after, expected).abs()) > math.cos(math.radians(7.0))


# --- fod_registration ------------------------------------------------------------


def test_fod_registration_validation():
    fixed = _fod_image((8, 8, 8), (4, 4, 4))
    moving = _fod_image((8, 8, 8), (3, 4, 5))
    with pytest.raises(ValueError, match="lmax"):
        fod_registration(fixed, moving, lmax=(0,), **FAST)
    with pytest.raises(ValueError, match="even"):
        fod_registration(fixed, moving, lmax=(0, 6), **FAST)
    scalar = ants.split_channels(fixed)[0]
    with pytest.raises(ValueError, match="component"):
        fod_registration(scalar, scalar, **FAST)
    with pytest.raises(ValueError, match="same number of spherical harmonic coefficients"):
        fod_registration(fixed, _fod_image((8, 8, 8), (4, 4, 4), lmax=2), **FAST)


@pytest.mark.parametrize("metric", ["lncc_joint", "mse"])
def test_fod_registration_reduces_loss_and_returns_vector_images(metric):
    fixed = _fod_image((14, 15, 16), (7, 7, 8))
    moving = _fod_image((14, 15, 16), (6, 8, 9))
    result = fod_registration(fixed, moving, syn_metric=metric, lmax=(2, 4), **FAST)
    history = result["loss_history"]
    assert len(history) == 10 and history[-1] < history[0]
    out = result["warpedmovout"]
    assert out.components == fixed.components and tuple(out.shape) == tuple(fixed.shape)
    assert result["warpedfixout"].components == fixed.components
    assert result["provenance"]["fod_lmax"] == (2, 4)
    assert result["provenance"]["reorientation"].startswith("FODReorientation(")
    assert np.isfinite(out.numpy()).all()


def test_fod_registration_defaults_and_affine_on_l0():
    fixed = _fod_image((12, 12, 12), (6, 6, 6))
    moving = _fod_image((12, 12, 12), (5, 6, 7))
    result = fod_registration(
        fixed, moving, type_of_transform="SyN", levels=(2, 1), reg_iterations=(3, 2), affine_iterations=(5, 5, 5)
    )
    provenance = result["provenance"]
    assert provenance["syn_metric"] == "lncc_joint"
    assert provenance["fod_lmax"] == (0, 2)
    assert provenance["output_interpolator"] == "cubic"
    assert result["affine_matrix"].shape == (3, 3)
    assert result["warpedmovout"].components == fixed.components
    assert result["fwdtransforms"][-1].endswith("0GenericAffine.mat")


def test_fod_registration_linear_only_reorients_by_the_affine():
    fixed = _fod_image((12, 12, 12), (6, 6, 6))
    moving = _fod_image((12, 12, 12), (5, 6, 7))
    result = fod_registration(
        fixed, moving, type_of_transform="Affine", affine_iterations=(5, 5, 5), affine_multi_start=False
    )
    assert result["warpedmovout"].components == fixed.components
    assert np.isfinite(result["warpedmovout"].numpy()).all() and result["provenance"]["fod_lmax"] is None


def test_extra_contrast_drives_the_registration_where_the_fods_carry_no_information():
    shape = (14, 15, 16)
    fixed = _fod_image(shape, (7, 7, 8))
    moving = fixed.clone()  # identical FODs: nothing for the SH channels to move
    kw = dict(syn_metric="lncc_joint", lmax=(2, 4), **dict(FAST, reg_iterations=(8, 6)))
    plain = fod_registration(fixed, moving, **kw)
    assert np.abs(ants.image_read(plain["fwdtransforms"][0]).numpy()).max() < 1e-4
    t1_fixed, t1_moving = _scalar_image(shape, (7, 7, 8)), _scalar_image(shape, (6, 8, 9))
    result = fod_registration(fixed, moving, extra_contrasts=[(t1_fixed, t1_moving)], **kw)
    warped_t1 = result["warpedmovout_extras"][0]
    before = np.mean((t1_moving.numpy() - t1_fixed.numpy()) ** 2)
    after = np.mean((warped_t1.numpy() - t1_fixed.numpy()) ** 2)
    assert after < 0.5 * before
    assert warped_t1.components == 1 and tuple(warped_t1.shape) == tuple(fixed.shape)
    assert result["warpedmovout"].components == fixed.components and np.isfinite(result["warpedmovout"].numpy()).all()
    assert np.abs(ants.image_read(result["fwdtransforms"][0]).numpy()).max() > 0.5
