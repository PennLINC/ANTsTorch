"""Tests for antstorch.syn.sh: MRtrix3-convention spherical harmonics and aPSF FOD reorientation."""
import math
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from antstorch.syn import sh


def _fibonacci_sphere(n: int) -> torch.Tensor:
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + 5.0**0.5) * i
    r = np.sqrt(1.0 - z * z)
    return torch.as_tensor(np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1))


def _two_fiber_fod(lmax: int, directions: torch.Tensor) -> torch.Tensor:
    """A sharp two-fiber FOD (sum of two aPSFs along bundled directions) with a positive l = 0 term."""
    apsf = sh.apsf_matrix(directions, lmax)
    fod = apsf[:, 3] + 0.6 * apsf[:, 17]
    fod[0] = fod[0] + 0.3
    return fod


def test_coefficient_counting_matches_mrtrix_convention():
    assert [sh.n_coefficients(l) for l in (0, 2, 4, 6, 8)] == [1, 6, 15, 28, 45]
    assert [sh.lmax_from_n(n) for n in (1, 6, 15, 28, 45)] == [0, 2, 4, 6, 8]
    assert sh.sh_index(0, 0) == 0 and sh.sh_index(2, -2) == 1 and sh.sh_index(2, 0) == 3 and sh.sh_index(4, 4) == 14
    with pytest.raises(ValueError):
        sh.lmax_from_n(10)
    with pytest.raises(ValueError):
        sh.n_coefficients(3)
    degrees = sh.coefficient_degrees(4)
    assert degrees.tolist() == [0] + [2] * 5 + [4] * 9


def test_sh_basis_is_orthonormal_and_has_known_values():
    lmax = 8
    points = _fibonacci_sphere(100_000)
    basis = sh.sh_basis(points, lmax)
    gram = (basis.T @ basis) * (4.0 * math.pi / points.shape[0])
    assert torch.allclose(gram, torch.eye(sh.n_coefficients(lmax), dtype=gram.dtype), atol=2e-4)
    # Y_00 = 1 / sqrt(4 pi) everywhere; Y_20 at the pole = sqrt(5 / (4 pi)).
    pole = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    values = sh.sh_basis(pole, 2)[0]
    assert values[0].item() == pytest.approx(1.0 / math.sqrt(4.0 * math.pi), rel=1e-9)
    assert values[sh.sh_index(2, 0)].item() == pytest.approx(math.sqrt(5.0 / (4.0 * math.pi)), rel=1e-9)
    # m != 0 terms vanish at the pole; the basis is finite for every direction, poles included.
    assert torch.allclose(values[[1, 2, 4, 5]], torch.zeros(4, dtype=torch.float64))
    assert torch.isfinite(sh.sh_basis(torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=torch.float64), 8)).all()


def test_sh_basis_m_sign_convention():
    # For a direction in the x-y plane with azimuth phi, index(l, m) carries cos(m phi) and index(l, -m) sin(m phi).
    phi = 0.7
    direction = torch.tensor([[math.cos(phi), math.sin(phi), 0.0]], dtype=torch.float64)
    values = sh.sh_basis(direction, 2)[0]
    ratio = values[sh.sh_index(2, -2)] / values[sh.sh_index(2, 2)]
    assert ratio.item() == pytest.approx(math.tan(2.0 * phi), rel=1e-9)


def test_bundled_direction_sets_load_as_unit_vectors():
    for n in (60, 300):
        directions = sh.load_directions(n, dtype=torch.float64)
        assert directions.shape == (n, 3)
        assert torch.allclose(directions.norm(dim=-1), torch.ones(n, dtype=torch.float64), atol=1e-12)
    with pytest.raises(ValueError):
        sh.load_directions(61)


def test_reorienter_identity_is_identity_and_rotation_rotates_peak():
    lmax = 8
    directions = sh.load_directions(60, dtype=torch.float64)
    reorienter = sh.FODReorienter(lmax, 60, dtype=torch.float64)
    fod = _two_fiber_fod(lmax, directions)
    same = reorienter.reorient_coefficients(fod[None], torch.eye(3, dtype=torch.float64))[0]
    assert torch.allclose(same, fod, atol=1e-10)
    # A rotation of the FOD moves its peak by that rotation.
    angle = math.radians(35.0)
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    rotated = reorienter.reorient_coefficients(fod[None], rotation)[0]
    probe = _fibonacci_sphere(20_000)
    amplitudes = sh.sh_basis(probe, lmax)
    peak_before = probe[int(torch.argmax(amplitudes @ fod))]
    peak_after = probe[int(torch.argmax(amplitudes @ rotated))]
    expected = rotation @ peak_before
    assert float(torch.dot(peak_after, expected).abs()) > math.cos(math.radians(2.0))
    # The rotation is orthogonal, so the FOD's total power is preserved (up to the aPSF least-squares fit).
    assert rotated.norm().item() == pytest.approx(fod.norm().item(), rel=2e-2)


def test_reorienter_modulation_scales_with_volume_change():
    # An isotropic scaling s I of the directions has |T d| / det T = 1 / s^2 for every aPSF,
    # so modulation multiplies the whole FOD by 1 / s^2 while the unmodulated FOD is unchanged.
    lmax = 4
    directions = sh.load_directions(60, dtype=torch.float64)
    reorienter = sh.FODReorienter(lmax, 60, dtype=torch.float64)
    fod = _two_fiber_fod(lmax, directions)
    scale = 1.3
    transform = scale * torch.eye(3, dtype=torch.float64)
    plain = reorienter.reorient_coefficients(fod[None], transform)[0]
    modulated = reorienter.reorient_coefficients(fod[None], transform, modulate=True)[0]
    assert torch.allclose(plain, fod, atol=1e-10)
    assert torch.allclose(modulated, fod / scale**2, atol=1e-10)


def test_reorient_image_matches_per_voxel_coefficients_and_skips_empty_voxels():
    lmax = 4
    directions = sh.load_directions(60, dtype=torch.float64)
    reorienter = sh.FODReorienter(lmax, 60, dtype=torch.float64)
    fod = _two_fiber_fod(lmax, directions)
    image = fod.view(1, -1, 1, 1, 1).repeat(1, 1, 3, 4, 5).clone()
    image[0, :, 0, 0, 0] = 0.0  # an empty voxel must be left untouched
    torch.manual_seed(0)
    transforms = torch.eye(3, dtype=torch.float64) + 0.2 * torch.randn(1, 3, 4, 5, 3, 3, dtype=torch.float64)
    out = reorienter.reorient(image, transforms, chunk_size=7)
    assert out.shape == image.shape
    assert torch.all(out[0, :, 0, 0, 0] == 0.0)
    expected = reorienter.reorient_coefficients(fod[None], transforms[0, 2, 3, 4][None])[0]
    assert torch.allclose(out[0, :, 2, 3, 4], expected, atol=1e-10)
    # gradients flow through the (checkpointed) chunks
    leaf = image.clone().requires_grad_(True)
    reorienter.reorient(leaf, transforms, chunk_size=7).sum().backward()
    assert torch.isfinite(leaf.grad).all()


def test_reorienter_rejects_too_few_directions():
    with pytest.raises(ValueError, match="directions"):
        sh.FODReorienter(8, torch.as_tensor(_fibonacci_sphere(30), dtype=torch.float64))
    with pytest.raises(ValueError):
        sh.FODReorienter(0, 60)


# --- Comparison against MRtrix3, when its binaries are available -------------

_MRTRIX_BIN = os.environ.get("MRTRIX_BIN") or (os.path.dirname(shutil.which("sh2amp")) if shutil.which("sh2amp") else None)
_needs_mrtrix = pytest.mark.skipif(_MRTRIX_BIN is None, reason="MRtrix3 binaries not found (set MRTRIX_BIN)")


@_needs_mrtrix
def test_sh_basis_matches_mrtrix_sh2amp(tmp_path):
    nib = pytest.importorskip("nibabel")
    lmax = 8
    rng = np.random.default_rng(0)
    coefficients = rng.normal(size=(3, 4, 5, sh.n_coefficients(lmax))).astype(np.float32)
    nib.save(nib.Nifti1Image(coefficients, np.diag([2.0, 2.0, 2.0, 1.0])), str(tmp_path / "sh.nii.gz"))
    directions_file = os.path.join(os.path.dirname(sh.__file__), "data", "directions_60.txt")
    subprocess.run(
        [os.path.join(_MRTRIX_BIN, "sh2amp"), str(tmp_path / "sh.nii.gz"), directions_file, str(tmp_path / "amp.nii.gz"), "-quiet"],
        check=True,
    )
    amplitudes = nib.load(str(tmp_path / "amp.nii.gz")).get_fdata()
    basis = sh.sh_basis(sh.load_directions(directions_file, dtype=torch.float64), lmax)
    expected = (torch.as_tensor(coefficients, dtype=torch.float64) @ basis.T).numpy()
    np.testing.assert_allclose(amplitudes, expected, atol=1e-4)


@_needs_mrtrix
@pytest.mark.parametrize("modulate", [False, True])
def test_linear_reorientation_matches_mrtransform(tmp_path, modulate):
    nib = pytest.importorskip("nibabel")
    lmax = 8
    directions_file = os.path.join(os.path.dirname(sh.__file__), "data", "directions_60.txt")
    directions = sh.load_directions(directions_file, dtype=torch.float64)
    fod = _two_fiber_fod(lmax, directions).numpy().astype(np.float32)
    shape = (5, 6, 7)
    constant = np.broadcast_to(fod, shape + (fod.size,)).copy()
    nib.save(nib.Nifti1Image(constant, np.diag([2.0, 2.0, 2.0, 1.0])), str(tmp_path / "fod.nii.gz"))
    # pull-back linear map (output -> input): rotation, anisotropic scale and shear
    angle = math.radians(30.0)
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    linear = rotation @ np.diag([1.15, 0.9, 1.05]) @ np.array([[1.0, 0.12, 0.0], [0.0, 1.0, 0.0], [-0.08, 0.0, 1.0]])
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    np.savetxt(str(tmp_path / "linear.txt"), matrix)
    command = [
        os.path.join(_MRTRIX_BIN, "mrtransform"), str(tmp_path / "fod.nii.gz"), "-linear", str(tmp_path / "linear.txt"),
        "-reorient_fod", "yes", "-directions", directions_file, str(tmp_path / "out.nii.gz"), "-quiet",
    ]
    if modulate:
        # mrtransform only modulates on its resampling path; resample onto the same grid and
        # compare the central voxel, which linear interpolation of a constant image leaves exact.
        command += ["-modulate", "fod", "-template", str(tmp_path / "fod.nii.gz"), "-interp", "linear"]
    subprocess.run(command, check=True)
    out = nib.load(str(tmp_path / "out.nii.gz")).get_fdata()
    reorienter = sh.FODReorienter(lmax, directions_file, dtype=torch.float64)
    transform = torch.as_tensor(np.linalg.inv(linear), dtype=torch.float64)
    expected = reorienter.reorient_coefficients(torch.as_tensor(fod, dtype=torch.float64)[None], transform, modulate=modulate)[0]
    np.testing.assert_allclose(out[2, 3, 3], expected.numpy(), atol=1e-5)
