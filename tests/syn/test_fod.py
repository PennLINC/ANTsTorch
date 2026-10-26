"""Tests for antstorch.syn.fod and FOD-aware syn_registration (multi-component spherical harmonic images)."""
import math

import ants
import numpy as np
import pytest
import torch

from antstorch.syn import sh, syn_registration
from antstorch.syn.bridge import ants_image_metadata, ants_image_to_tensor, tensor_to_ants_image
from antstorch.syn.core.losses import local_ncc_loss_nd
from antstorch.syn.fod import (
    FODRegistration,
    default_fod_lmax,
    direction_transform,
    fod_image_read,
    fod_image_write,
    pullback_jacobian,
    sh_lmax_of_image,
)

_DIRECTION = np.diag([-1.0, -1.0, 1.0])  # what ITK reports for a plain RAS NIfTI


def _blob(shape, center, sigma):
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float64)
    return np.exp(-(((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) / (2.0 * sigma**2)))


def _fod_image(shape, center, lmax=4, direction_index=3, spacing=(2.0, 2.0, 2.0), direction=_DIRECTION):
    """A Gaussian blob carrying one aPSF fiber everywhere (channel-last (x, y, z, n) ANTs vector image)."""
    directions = sh.load_directions(60, dtype=torch.float64)
    fiber = sh.apsf_matrix(directions, lmax)[:, direction_index].numpy()
    weights = _blob(shape, center, 3.0)
    array = weights[..., None] * fiber[None, None, None, :]
    array[..., 0] += 0.05 * weights
    array = np.ascontiguousarray(np.transpose(array, (2, 1, 0, 3)).astype(np.float32))
    return ants.from_numpy(array, origin=(-20.0, -18.0, 5.0), spacing=spacing, direction=direction, has_components=True)


# --- bridge -----------------------------------------------------------------


def test_multicomponent_bridge_roundtrip_and_channel_order():
    image = _fod_image((6, 7, 8), (3, 3, 4))
    tensor = ants_image_to_tensor(image, normalize=False)
    assert tensor.shape == (1, image.components) + tuple(reversed(image.shape))
    # channel c of the tensor is component c of the image, with (z, y, x) spatial order
    np.testing.assert_allclose(tensor[0, 2].numpy(), image.numpy()[..., 2].transpose(2, 1, 0), atol=1e-6)
    recovered = tensor_to_ants_image(tensor, image)
    assert recovered.components == image.components
    np.testing.assert_allclose(recovered.numpy(), image.numpy(), atol=1e-6)
    assert ants_image_metadata(recovered)["spacing"] == ants_image_metadata(image)["spacing"]


def test_fod_image_read_write_roundtrip_keeps_coefficients_and_geometry(tmp_path):
    pytest.importorskip("nibabel")
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


# --- degree schedule ----------------------------------------------------------


def test_default_fod_lmax_follows_mrregister():
    assert default_fod_lmax(3, 8) == (0, 2, 4)
    assert default_fod_lmax(4, 8) == (0, 2, 4, 4)
    assert default_fod_lmax(3, 2) == (0, 2, 2)
    assert default_fod_lmax(2, 8, max_lmax=8) == (0, 2)


# --- Jacobians and frames -----------------------------------------------------


def test_pullback_jacobian_recovers_an_affine_field_with_rotated_direction():
    # u(x) = (A - I) x + b in physical LPS coordinates: the Jacobian of x + u must be A everywhere inside.
    torch.manual_seed(0)
    angle = math.radians(20.0)
    direction = np.array([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    direction = direction @ np.diag([-1.0, 1.0, 1.0])
    spacing = (1.5, 2.0, 2.5)
    origin = (3.0, -4.0, 5.0)
    shape_zyx = (7, 8, 9)
    from antstorch.syn.core.grid import get_physical_grid_torch

    X = get_physical_grid_torch(shape_zyx, spacing, origin, direction, dtype=torch.float64)  # (1, z, y, x, 3) zyx comps
    A_xyz = torch.eye(3, dtype=torch.float64) + 0.1 * torch.randn(3, 3, dtype=torch.float64)
    A_zyx = A_xyz.flip(0).flip(1)
    u = X @ (A_zyx - torch.eye(3, dtype=torch.float64)).T + torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    jac = pullback_jacobian(u, spacing, direction)
    assert jac.shape == (1,) + shape_zyx + (3, 3)
    inner = jac[0, 1:-1, 1:-1, 1:-1]
    assert torch.allclose(inner, A_xyz.expand_as(inner), atol=1e-9)
    # an extra affine (reversed component order) multiplies from the left
    M_zyx = (torch.eye(3, dtype=torch.float64) + 0.05 * torch.randn(3, 3, dtype=torch.float64))
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


def test_fod_registration_hook_rotates_fibers_like_the_warp():
    """A pure rotation of the fixed-to-midpoint half-warp must rotate the FOD peaks by the same rotation (in RAS)."""
    lmax = 4
    image = _fod_image((12, 12, 12), (6, 6, 6), lmax=lmax)
    meta = ants_image_metadata(image)
    from antstorch.syn.core.grid import get_physical_grid_torch

    X = get_physical_grid_torch(meta["torch_shape"], meta["spacing"], meta["origin"], meta["direction"], dtype=torch.float64)
    angle = math.radians(40.0)
    R_lps = torch.tensor(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    R_zyx = R_lps.flip(0).flip(1)
    u = X @ (R_zyx - torch.eye(3, dtype=torch.float64)).T  # pull-back phi(x) = R x
    I = ants_image_to_tensor(image, dtype=torch.float64, normalize=False)
    registration = FODRegistration(lmax, (lmax,), dtype=torch.float64)
    hook = registration.midpoint_hook(lmax, meta, torch.eye(3, dtype=torch.float64))
    out, _ = hook(I, I, u, u)
    probe = sh.load_directions(300, dtype=torch.float64)
    basis = sh.sh_basis(probe, lmax)
    center = (6, 6, 6)
    peak_in = probe[int(torch.argmax(basis @ I[0, :, center[0], center[1], center[2]]))]
    peak_out = probe[int(torch.argmax(basis @ out[0, :, center[0], center[1], center[2]]))]
    # directions live in RAS; the LPS rotation R maps to flip R flip, and the fiber moves by its inverse
    flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))
    expected = flip @ torch.linalg.inv(R_lps) @ flip @ peak_in
    assert float(torch.dot(peak_out, expected).abs()) > math.cos(math.radians(7.0))  # 300-direction probe resolution
    assert registration.midpoint_hook(0, meta, torch.eye(3, dtype=torch.float64)) is None


# --- losses -------------------------------------------------------------------


def test_joint_lncc_reduces_to_lncc_for_one_channel_and_differs_for_many():
    torch.manual_seed(0)
    I = torch.rand(1, 1, 9, 10, 11)
    J = torch.rand(1, 1, 9, 10, 11)
    single = local_ncc_loss_nd(I, J, window_size=3)
    joint = local_ncc_loss_nd(I, J, window_size=3, joint_channels=True)
    assert torch.allclose(single, joint)
    I3 = torch.cat([I, 0.01 * torch.rand(1, 2, 9, 10, 11)], dim=1)
    J3 = torch.cat([J, 0.01 * torch.rand(1, 2, 9, 10, 11)], dim=1)
    per_channel = local_ncc_loss_nd(I3, J3, window_size=3)
    joint3 = local_ncc_loss_nd(I3, J3, window_size=3, joint_channels=True)
    assert not torch.allclose(per_channel, joint3)
    assert -1.0 <= float(joint3) <= 0.0
    with pytest.raises(ValueError):
        local_ncc_loss_nd(I3, J3, window_size=3, joint_channels=True, use_ants_pseudo_gradient=True)


# --- syn_registration(fod=True) -----------------------------------------------


def test_syn_registration_fod_validation():
    fixed = _fod_image((8, 8, 8), (4, 4, 4))
    moving = _fod_image((8, 8, 8), (3, 4, 5))
    with pytest.raises(ValueError, match="fod_lmax"):
        syn_registration(fixed, moving, type_of_transform="SyNOnly", fod=True, levels=(2, 1), reg_iterations=(1, 1), fod_lmax=(0,))
    with pytest.raises(ValueError):
        syn_registration(fixed, moving, type_of_transform="SyNOnly", fod=True, levels=(2, 1), reg_iterations=(1, 1), fod_lmax=(0, 6))
    scalar = ants.split_channels(fixed)[0]
    with pytest.raises(ValueError):
        syn_registration(scalar, scalar, type_of_transform="SyNOnly", fod=True, levels=(1,), reg_iterations=(1,))


@pytest.mark.parametrize("metric", ["lncc_joint", "mse"])
def test_syn_registration_fod_reduces_loss_and_returns_vector_images(metric):
    fixed = _fod_image((14, 15, 16), (7, 7, 8))
    moving = _fod_image((14, 15, 16), (6, 8, 9))
    result = syn_registration(
        fixed, moving, type_of_transform="SyNOnly", fod=True, syn_metric=metric,
        levels=(2, 1), reg_iterations=(6, 4), fod_lmax=(2, 4), device="cpu",
    )
    history = result["loss_history"]
    assert len(history) == 10 and history[-1] < history[0]
    out = result["warpedmovout"]
    assert out.components == fixed.components and tuple(out.shape) == tuple(fixed.shape)
    assert result["warpedfixout"].components == fixed.components
    assert result["provenance"]["fod"] is True and result["provenance"]["fod_lmax"] == (2, 4)
    assert np.isfinite(out.numpy()).all()


def test_syn_registration_fod_default_schedule_metric_and_affine_on_l0():
    fixed = _fod_image((12, 12, 12), (6, 6, 6))
    moving = _fod_image((12, 12, 12), (5, 6, 7))
    result = syn_registration(
        fixed, moving, type_of_transform="SyN", fod=True, levels=(2, 1), reg_iterations=(3, 2),
        affine_iterations=(5, 5, 5), device="cpu",
    )
    provenance = result["provenance"]
    assert provenance["syn_metric"] == "lncc_joint"
    assert provenance["fod_lmax"] == (0, 2)
    assert result["affine_matrix"].shape == (3, 3)
    assert result["warpedmovout"].components == fixed.components
    assert result["fwdtransforms"][-1].endswith("0GenericAffine.mat")
