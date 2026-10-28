"""Tests for FOD (spherical harmonic) template construction with build_template(fod=True)."""
import os

import ants
import numpy as np
import pytest
import torch

from antstorch.ants_transform_io import write_affine_transform
from antstorch.syn import build_template, sh
from antstorch.syn.fod import FODRegistration
from antstorch.syn.template import _initial_template, _resample_vector_onto, _sharpen_vector_image, _warp_sh_template

_DIRECTION = np.diag([-1.0, -1.0, 1.0])


def _blob(shape, center, sigma):
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float64)
    return np.exp(-(((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) / (2.0 * sigma**2)))


def _fod_image(shape, center, lmax=4, direction_index=3, origin=(-20.0, -18.0, 5.0)):
    directions = sh.load_directions(60, dtype=torch.float64)
    fiber = sh.apsf_matrix(directions, lmax)[:, direction_index].numpy()
    # A large blob plus a smaller satellite: an asymmetric shape, so that the affine
    # stage cannot land on a reflection (a lone Gaussian blob is mirror symmetric).
    satellite = tuple(c + o for c, o in zip(center, (2.5, -2.0, 1.5)))
    weights = _blob(shape, center, 3.0) + 0.6 * _blob(shape, satellite, 1.5)
    array = weights[..., None] * fiber[None, None, None, :]
    array[..., 0] += 0.05 * weights
    array = np.ascontiguousarray(np.transpose(array, (2, 1, 0, 3)).astype(np.float32))
    return ants.from_numpy(array, origin=origin, spacing=(2.0, 2.0, 2.0), direction=_DIRECTION, has_components=True)


def _smooth_warp(reference, amplitude=1.5):
    """A smooth synthetic displacement field (vector ANTsImage on ``reference``'s grid, ITK (x, y, z) components)."""
    x, y, z = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in reference.shape], indexing="ij")
    field = np.stack(
        [amplitude * np.sin(x / 4.0) * np.cos(y / 5.0), -amplitude * np.cos(z / 3.0), amplitude * np.sin((x + y) / 6.0)], axis=-1
    ).astype(np.float32)
    return ants.from_numpy(field, origin=reference.origin, spacing=reference.spacing, direction=reference.direction, has_components=True)


def test_warp_sh_template_matches_ants_apply_transforms_for_a_scalar_image(tmp_path):
    """With one channel there is nothing to reorient, so the sampling must agree with ants.apply_transforms."""
    rng = np.random.default_rng(0)
    scalar = ants.from_numpy(
        np.ascontiguousarray(_blob((14, 15, 16), (7, 8, 7), 3.0).transpose(2, 1, 0).astype(np.float32) + 0.1 * rng.random((16, 15, 14), dtype=np.float32)),
        origin=(-20.0, -18.0, 5.0), spacing=(2.0, 2.0, 2.0), direction=_DIRECTION,
    )
    # ANTsPy has no 1-component vector image; duplicate the scalar into two channels (lmax 0: nothing to reorient).
    vector = ants.from_numpy(
        np.ascontiguousarray(np.stack([scalar.numpy(), scalar.numpy()], axis=-1)),
        origin=scalar.origin, spacing=scalar.spacing, direction=scalar.direction, has_components=True,
    )
    warp = _smooth_warp(scalar)
    warp_fn = str(tmp_path / "warp.nii.gz")
    ants.image_write(warp, warp_fn)
    angle = np.radians(8.0)
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.05]])
    translation = np.array([1.0, -2.0, 0.5])
    affine_fn = str(tmp_path / "affine.mat")
    write_affine_transform(torch.as_tensor(matrix), torch.as_tensor(translation), 3, affine_fn)
    expected = ants.apply_transforms(fixed=scalar, moving=scalar, transformlist=[warp_fn, affine_fn], whichtoinvert=[0, 1]).numpy()
    tools = FODRegistration(0, (0,), device="cpu")  # lmax 0: sampling only, no reorientation
    got = _warp_sh_template(vector, warp, affine_fn, tools)
    assert got.components == vector.components
    inner = (slice(2, -2),) * 3
    np.testing.assert_allclose(got.numpy()[..., 0][inner], expected[inner], atol=2e-3)
    np.testing.assert_allclose(got.numpy()[..., 1][inner], expected[inner], atol=2e-3)
    # warp only (no affine) and affine only must also agree
    expected_w = ants.apply_transforms(fixed=scalar, moving=scalar, transformlist=[warp_fn], whichtoinvert=[0]).numpy()
    got_w = _warp_sh_template(vector, warp, None, tools).numpy()[..., 0]
    np.testing.assert_allclose(got_w[inner], expected_w[inner], atol=2e-3)
    expected_a = ants.apply_transforms(fixed=scalar, moving=scalar, transformlist=[affine_fn], whichtoinvert=[1]).numpy()
    got_a = _warp_sh_template(vector, None, affine_fn, tools).numpy()[..., 0]
    np.testing.assert_allclose(got_a[inner], expected_a[inner], atol=2e-3)


def test_warp_sh_template_reorients_fibers_with_the_affine():
    """A pure rotation about z as the average affine must rotate every FOD's peak (in RAS) by its inverse."""
    lmax = 4
    image = _fod_image((12, 12, 12), (6, 6, 6), lmax=lmax)
    angle = np.radians(30.0)
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        affine_fn = os.path.join(d, "affine.mat")
        write_affine_transform(torch.as_tensor(matrix), torch.zeros(3, dtype=torch.float64), 3, affine_fn)
        tools = FODRegistration(lmax, (lmax,), device="cpu")
        out = _warp_sh_template(image, None, affine_fn, tools)
    probe = sh.load_directions(300, dtype=torch.float64)
    basis = sh.sh_basis(probe, lmax)
    before = torch.as_tensor(image.numpy()[6, 6, 6], dtype=torch.float64)
    # the template is sampled at A^-1 x: pick the voxel that maps to the blob center... the blob is wide, use the center
    after = torch.as_tensor(out.numpy()[6, 6, 6], dtype=torch.float64)
    peak_before = probe[int(torch.argmax(basis @ before))]
    peak_after = probe[int(torch.argmax(basis @ after))]
    # pull-back phi(x) = A^-1 x, directions move by (d phi / dx)^-1 = A, conjugated LPS -> RAS
    flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))
    expected = flip @ torch.as_tensor(matrix, dtype=torch.float64) @ flip @ peak_before
    assert float(torch.dot(peak_after, expected).abs()) > np.cos(np.radians(7.0))


def test_sharpen_vector_image_matches_per_channel_scalar_sharpening():
    image = _fod_image((10, 11, 12), (5, 5, 6))
    out = _sharpen_vector_image(image, 0.75)
    assert out.components == image.components and tuple(out.shape) == tuple(image.shape)
    channel = ants.split_channels(image)[0]
    expected = channel * 0.75 + ants.iMath(channel, "Sharpen") * 0.25
    np.testing.assert_allclose(out.numpy()[..., 0], expected.numpy(), atol=1e-5)
    np.testing.assert_allclose(np.asarray(out.direction), np.asarray(image.direction))


@pytest.mark.parametrize("interpolator", ["linear", "cubic"])
def test_resample_vector_onto_identity_and_shifted_grid(interpolator):
    image = _fod_image((10, 11, 12), (5, 5, 6))
    same = _resample_vector_onto(image, image, interpolator=interpolator)
    np.testing.assert_allclose(same.numpy(), image.numpy(), atol=1e-6)
    # a copy of the grid (not the same object) still reproduces the voxel values with an interpolating kernel
    copy = ants.from_numpy(image.numpy() * 0.0, origin=image.origin, spacing=image.spacing, direction=image.direction, has_components=True)
    np.testing.assert_allclose(_resample_vector_onto(image, copy, interpolator=interpolator).numpy(), image.numpy(), atol=1e-5)
    shifted = _fod_image((10, 11, 12), (5, 5, 6), origin=(-19.0, -17.0, 6.0))
    out = _resample_vector_onto(image, shifted, interpolator=interpolator)
    if interpolator == "cubic":
        # closer to the analytic shifted image than trilinear resampling (the inputs are smooth blobs)
        # the target grid's origin is 1 mm further along each LPS axis: with direction diag(-1, -1, 1) and
        # 2 mm voxels, target voxel (i, j, k) samples the source at (i - 0.5, j - 0.5, k + 0.5), so the blob
        # (centered in (z, y, x) index order) moves to (cz - 0.5, cy + 0.5, cx + 0.5)
        truth = _fod_image((10, 11, 12), (4.5, 5.5, 6.5)).numpy()
        linear = _resample_vector_onto(image, shifted, interpolator="linear").numpy()
        inner = (slice(2, -2),) * 3
        assert np.abs(out.numpy() - truth)[inner].mean() < np.abs(linear - truth)[inner].mean()
        return
    expected = ants.resample_image_to_target(ants.split_channels(image)[0], ants.split_channels(shifted)[0], interp_type="linear").numpy()
    inner = (slice(1, -1),) * 3
    np.testing.assert_allclose(out.numpy()[..., 0][inner], expected[inner], atol=1e-4)


def test_warp_sh_template_scalar_cubic_identity_and_rotation(tmp_path):
    """Without a transform the cubic path returns the template unchanged; with ``fod_tools=None`` nothing is reoriented."""
    rng = np.random.default_rng(1)
    scalar = ants.from_numpy(
        np.ascontiguousarray(rng.random((12, 11, 10), dtype=np.float32)), origin=(-20.0, -18.0, 5.0), spacing=(2.0, 2.0, 2.0), direction=_DIRECTION
    )
    identity_fn = str(tmp_path / "identity.mat")
    write_affine_transform(torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 3, identity_fn)
    zero_warp = ants.from_numpy(np.zeros(scalar.shape + (3,), dtype=np.float32), origin=scalar.origin, spacing=scalar.spacing, direction=scalar.direction, has_components=True)
    for interpolator in ("linear", "cubic", "nearest"):
        got = _warp_sh_template(scalar, zero_warp, identity_fn, None, interpolator)
        np.testing.assert_allclose(got.numpy(), scalar.numpy(), atol=1e-5)
    # the approximating B-spline does not reproduce the voxel values
    assert np.abs(_warp_sh_template(scalar, zero_warp, identity_fn, None, "bspline").numpy() - scalar.numpy()).max() > 1e-2
    # a translation by one voxel is exact for an interpolating kernel (away from the borders)
    shift_fn = str(tmp_path / "shift.mat")
    write_affine_transform(torch.eye(3, dtype=torch.float64), torch.tensor([-2.0, 0.0, 0.0], dtype=torch.float64), 3, shift_fn)
    got = _warp_sh_template(scalar, None, shift_fn, None, "cubic").numpy()
    np.testing.assert_allclose(got[1:, :, :], scalar.numpy()[:-1, :, :], atol=1e-5)


def test_build_template_fod_runs_and_returns_a_vector_template(tmp_path):
    images = [_fod_image((12, 13, 14), c) for c in ((6, 6, 7), (5, 7, 8), (7, 6, 6))]
    # A plain gradient-descent affine (no multi-start): the multi-start search can pick a
    # 180-degree rotated optimum on these nearly symmetric toy shapes, and ANTs' affine
    # averaging then decomposes it into negative scales.
    result = build_template(
        image_list=images, iterations=2, gradient_step=0.2, blending_weight=0.75, output_dir=str(tmp_path),
        type_of_transform="SyN", fod=True, levels=(2, 1), reg_iterations=(3, 2), affine_iterations=(10, 10, 10),
        affine_multi_start=False, device="cpu",
    )
    template = result["template"]
    assert template.components == images[0].components and tuple(template.shape) == tuple(images[0].shape)
    assert np.isfinite(template.numpy()).all()
    assert result["n_iterations"] == 2 and len(result["convergence"]) == 2
    assert all(r["l2_norm"] >= 0.0 for r in result["shape_residuals"])
    assert all(w.components == images[0].components for w in result["warped_images"])
    assert os.path.exists(os.path.join(str(tmp_path), "avgWarp_0.nii.gz"))
    with pytest.raises(ValueError, match="output_interpolator"):
        build_template(image_list=images, iterations=1, output_dir=str(tmp_path), fod=True, levels=(1,), reg_iterations=(1,), device="cpu", output_interpolator="sinc")


def test_build_template_scalar_cubic_template_update(tmp_path):
    images = [ants.split_channels(_fod_image((12, 13, 14), c))[0] for c in ((6, 6, 7), (5, 7, 8), (7, 6, 6))]
    result = build_template(
        image_list=images, iterations=1, gradient_step=0.2, blending_weight=1.0, output_dir=str(tmp_path),
        type_of_transform="SyN", levels=(2, 1), reg_iterations=(3, 2), affine_iterations=(10, 10, 10), affine_multi_start=False,
        device="cpu", output_interpolator="cubic",
    )
    assert result["template"].components == 1 and np.isfinite(result["template"].numpy()).all()
    assert tuple(result["template"].shape) == tuple(images[0].shape)


def test_build_template_fod_rejects_mismatched_components(tmp_path):
    images = [_fod_image((8, 8, 8), (4, 4, 4), lmax=4), _fod_image((8, 8, 8), (4, 4, 4), lmax=2)]
    with pytest.raises(ValueError, match="same number"):
        build_template(image_list=images, iterations=1, output_dir=str(tmp_path), fod=True, levels=(1,), reg_iterations=(1,), device="cpu")


def _scalar_blob(shape, center, sigma=2.5, spacing=(2.0, 2.0, 2.0), origin=(-20.0, -18.0, 5.0)):
    array = np.ascontiguousarray(np.transpose(_blob(shape, center, sigma), (2, 1, 0)).astype(np.float32))
    return ants.from_numpy(array, origin=origin, spacing=spacing, direction=_DIRECTION)


def test_build_template_with_an_extra_contrast(tmp_path):
    centers = ((6, 6, 7), (5, 7, 8), (7, 6, 6))
    images = [_fod_image((12, 13, 14), c) for c in centers]
    extras = [_scalar_blob((12, 13, 14), c) * 3.0 for c in centers]
    common = dict(
        gradient_step=0.2, output_dir=str(tmp_path), type_of_transform="SyN", fod=True, levels=(2, 1), reg_iterations=(3, 2),
        affine_iterations=(10, 10, 10), affine_multi_start=False, device="cpu",
    )
    result = build_template(
        image_list=images, extra_contrasts=[extras], extra_contrast_weights=[0.5], iterations=2, blending_weight=0.75, **common
    )
    assert result["template"].components == images[0].components
    assert len(result["extra_templates"]) == 1
    extra_template = result["extra_templates"][0]
    assert extra_template.components == 1 and tuple(extra_template.shape) == tuple(images[0].shape)
    assert np.allclose(extra_template.spacing, images[0].spacing) and np.isfinite(extra_template.numpy()).all()
    # The extra template is an average of the warped extras: its peak sits between the inputs' peaks.
    assert 0.5 * 3.0 < extra_template.numpy().max() <= 3.0 * 1.05
    assert len(result["warped_extra_images"]) == 1 and len(result["warped_extra_images"][0]) == 3
    assert all(w.components == 1 and tuple(w.shape) == tuple(images[0].shape) for w in result["warped_extra_images"][0])
    assert result["n_iterations"] == 2

    # Extras on a finer grid are resampled onto the primary grid first.
    fine = [_scalar_blob((24, 26, 28), tuple(2 * v for v in c), sigma=5.0, spacing=(1.0, 1.0, 1.0)) for c in centers]
    fine_result = build_template(image_list=images, extra_contrasts=[fine], iterations=1, blending_weight=1.0, **common)
    assert tuple(fine_result["extra_templates"][0].shape) == tuple(images[0].shape)
    assert np.allclose(fine_result["extra_templates"][0].spacing, images[0].spacing)

    with pytest.raises(ValueError, match="extra_contrasts\\[0\\]"):
        build_template(image_list=images, extra_contrasts=[extras[:2]], iterations=1, **common)
    with pytest.raises(ValueError, match="extra_contrast_weights"):
        build_template(image_list=images, extra_contrasts=[extras], extra_contrast_weights=[1.0, 1.0], iterations=1, **common)
    with pytest.raises(ValueError, match="scalar"):
        build_template(image_list=images, extra_contrasts=[images], iterations=1, **common)


def test_build_template_without_extra_contrasts_returns_empty_extra_lists(tmp_path):
    images = [ants.split_channels(_fod_image((10, 10, 10), c))[0] for c in ((5, 5, 5), (4, 5, 6))]
    result = build_template(
        image_list=images, iterations=1, output_dir=str(tmp_path), type_of_transform="SyNOnly", levels=(1,), reg_iterations=(1,), device="cpu"
    )
    assert result["extra_templates"] == [] and result["warped_extra_images"] == []


def test_build_template_continues_from_previous_templates(tmp_path):
    centers = ((6, 6, 7), (5, 7, 8), (7, 6, 6))
    images = [_fod_image((12, 13, 14), c) for c in centers]
    extras = [_scalar_blob((12, 13, 14), c) * 3.0 for c in centers]
    common = dict(
        gradient_step=0.2, type_of_transform="SyN", fod=True, levels=(2, 1), reg_iterations=(3, 2), affine_iterations=(10, 10, 10),
        affine_multi_start=False, device="cpu", blending_weight=1.0,
    )
    first = build_template(image_list=images, extra_contrasts=[extras], iterations=1, output_dir=str(tmp_path / "first"), **common)
    second = build_template(
        image_list=images, extra_contrasts=[extras], iterations=1, output_dir=str(tmp_path / "second"),
        initial_template=first["template"], initial_extra_templates=first["extra_templates"], **common,
    )
    assert second["n_iterations"] == 1
    extra_template = second["extra_templates"][0]
    assert extra_template.components == 1 and tuple(extra_template.shape) == tuple(first["template"].shape)
    assert np.isfinite(extra_template.numpy()).all()
    # One more iteration refines the extra template; it stays an average of the inputs rather than restarting from scratch.
    assert 0.5 * 3.0 < extra_template.numpy().max() <= 3.0 * 1.05
    assert np.abs(extra_template.numpy() - first["extra_templates"][0].numpy()).max() < 0.5 * 3.0
    with pytest.raises(ValueError, match="initial_extra_templates must hold one image per extra contrast"):
        build_template(
            image_list=images, extra_contrasts=[extras], iterations=1, output_dir=str(tmp_path / "bad"),
            initial_template=first["template"], initial_extra_templates=[], **common,
        )


def test_build_template_initial_alignment_starts_from_aligned_inputs(tmp_path):
    centers = ((6, 6, 7), (4, 7, 8), (8, 6, 6))
    images = [_fod_image((12, 13, 14), c) for c in centers]
    extras = [_scalar_blob((12, 13, 14), c) * 3.0 for c in centers]
    common = dict(
        gradient_step=0.2, type_of_transform="SyN", fod=True, levels=(2, 1), reg_iterations=(2, 2), affine_iterations=(10, 10, 10),
        affine_multi_start=False, device="cpu", blending_weight=1.0,
    )
    res = build_template(
        image_list=images, extra_contrasts=[extras], iterations=1, output_dir=str(tmp_path / "aligned"), initial_alignment="Rigid", **common
    )
    pre = res["initial_alignment"]
    assert pre is not None and pre["n_iterations"] == 1 and res["n_iterations"] == 1
    assert os.path.isfile(os.path.join(tmp_path, "aligned", "initial_rigid", "iter00_img0000", "out0GenericAffine.mat"))
    assert np.isfinite(res["template"].numpy()).all() and np.isfinite(res["extra_templates"][0].numpy()).all()
    # Rigidly aligning the blobs before averaging gives a sharper starting template than the plain average.
    plain = _initial_template(extras, [1.0 / 3.0] * 3, "linear")
    assert pre["extra_templates"][0].numpy().max() > plain.numpy().max() + 0.1 * 3.0
    # A supplied initial template skips the pre-alignment.
    again = build_template(
        image_list=images, extra_contrasts=[extras], iterations=1, output_dir=str(tmp_path / "given"), initial_alignment="Rigid",
        initial_template=res["template"], initial_extra_templates=res["extra_templates"], **common,
    )
    assert again["initial_alignment"] is None
    with pytest.raises(ValueError, match="initial_alignment must be one of"):
        build_template(image_list=images, iterations=1, output_dir=str(tmp_path / "bad"), initial_alignment="SyN", **common)
