"""Tests for ``antstorch.syn.build_template`` (port of ``syntx.build_template``).

CPU-only, tiny 2D phantoms (32x32) with short SyN schedules for speed.
"""

import os

import ants
import numpy as np
import pytest
import torch

from antstorch.ants_transform_io import write_affine_transform
from antstorch.syn import build_template
from antstorch.syn.template import (
    _apply_transforms_torch,
    _compute_shape_residual,
    _initial_template,
    _normalize_weights,
    _resample_vector_onto,
    _sharpen_image,
    _template_change,
)

FAST = dict(levels=(2, 1), reg_iterations=(6, 3))


def _disc(cx, cy, size=32, radius=6):
    yy, xx = np.mgrid[0:size, 0:size]
    arr = (((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2).astype(np.float32)
    return ants.from_numpy(arr)


def _cohort(offsets=((-3, 0), (3, 0), (0, -3), (0, 3)), size=32):
    c = size // 2
    return [_disc(c + dx, c + dy, size) for dx, dy in offsets]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestNormalizeWeights:
    def test_uniform_default(self):
        np.testing.assert_allclose(_normalize_weights(None, 4), [0.25] * 4)

    def test_normalised_to_one(self):
        w = _normalize_weights([1, 1, 2], 3)
        assert w.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(w, [0.25, 0.25, 0.5])

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="len\\(weights\\)"):
            _normalize_weights([1, 2], 3)

    def test_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            _normalize_weights([1, -1, 1], 3)

    def test_zero_sum(self):
        with pytest.raises(ValueError, match="positive sum"):
            _normalize_weights([0, 0], 2)


class TestInitialTemplate:
    def test_uniform_mean(self):
        imgs = _cohort()
        got = _initial_template(imgs, _normalize_weights(None, 4))
        expected = np.mean([i.numpy() for i in imgs], axis=0)
        np.testing.assert_allclose(got.numpy(), expected, atol=1e-6)

    def test_weights_select_image(self):
        imgs = _cohort()
        got = _initial_template(imgs, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(got.numpy(), imgs[0].numpy(), atol=1e-6)


class TestTemplateChange:
    def test_identical_is_zero(self):
        img = _disc(16, 16)
        assert _template_change(img, img) == 0.0

    def test_known_value(self):
        a = ants.from_numpy(np.zeros((4, 4), dtype=np.float32))
        b = ants.from_numpy(np.full((4, 4), 0.5, dtype=np.float32))
        assert _template_change(a, b) == pytest.approx(0.5)


class TestShapeResidual:
    def _field(self, arr, spacing=(1.0, 1.0)):
        return ants.from_numpy(arr.astype(np.float32), spacing=spacing, has_components=True)

    def test_zero_field(self):
        sr = _compute_shape_residual(self._field(np.zeros((8, 8, 2))))
        assert sr == {"l2_norm": 0.0, "membrane_energy": 0.0, "bending_energy": 0.0}

    def test_constant_translation(self):
        arr = np.zeros((8, 8, 2))
        arr[..., 0] = 3.0
        arr[..., 1] = 4.0
        sr = _compute_shape_residual(self._field(arr))
        assert sr["l2_norm"] == pytest.approx(5.0)
        assert sr["membrane_energy"] == pytest.approx(0.0, abs=1e-9)
        assert sr["bending_energy"] == pytest.approx(0.0, abs=1e-9)

    def test_linear_ramp_has_membrane_but_no_bending(self):
        xx = np.tile(np.arange(16, dtype=np.float64), (16, 1))
        arr = np.stack([xx, np.zeros_like(xx)], axis=-1)
        sr = _compute_shape_residual(self._field(arr))
        assert sr["membrane_energy"] == pytest.approx(1.0)
        assert sr["bending_energy"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


class TestErrors:
    def test_empty_image_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_template(image_list=[])

    def test_missing_image_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_template()

    @pytest.mark.parametrize("bw", [0.0, -0.1, 1.5])
    def test_bad_blending_weight(self, bw):
        with pytest.raises(ValueError, match="blending_weight"):
            build_template(image_list=_cohort(), blending_weight=bw)

    @pytest.mark.parametrize("gs", [-0.1, 1.1])
    def test_bad_gradient_step(self, gs):
        with pytest.raises(ValueError, match="gradient_step"):
            build_template(image_list=_cohort(), gradient_step=gs)

    def test_bad_iterations(self):
        with pytest.raises(ValueError, match="iterations"):
            build_template(image_list=_cohort(), iterations=0)

    def test_bad_weights_length(self):
        with pytest.raises(ValueError, match="len\\(weights\\)"):
            build_template(image_list=_cohort(), weights=[1, 2])

    @pytest.mark.parametrize("name", ["initial_affine", "outprefix"])
    def test_managed_kwargs_rejected(self, name):
        with pytest.raises(TypeError, match=name):
            build_template(image_list=_cohort(), **{name: None})


# ---------------------------------------------------------------------------
# integration (runs syn_registration)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def syn_result(tmp_path_factory):
    work = tmp_path_factory.mktemp("tmpl")
    out = build_template(
        image_list=_cohort(), iterations=2, output_dir=str(work), blending_weight=0.75, **FAST
    )
    return out, work


class TestBuildTemplateSyN:
    def test_schema(self, syn_result):
        out, _ = syn_result
        assert set(out) == {
            "template",
            "warped_images",
            "extra_templates",
            "warped_extra_images",
            "fwdtransforms",
            "invtransforms",
            "convergence",
            "shape_residuals",
            "n_iterations",
            "weights",
            "work_dir",
            "initial_alignment",
            "elapsed_sec",
        }

    def test_template_geometry(self, syn_result):
        out, _ = syn_result
        assert isinstance(out["template"], ants.ANTsImage)
        assert out["template"].shape == (32, 32)
        assert np.isfinite(out["template"].numpy()).all()

    def test_iteration_bookkeeping(self, syn_result):
        out, _ = syn_result
        assert out["n_iterations"] == 2
        assert len(out["convergence"]) == 2
        assert len(out["shape_residuals"]) == 2
        assert all(np.isfinite(c) and c >= 0 for c in out["convergence"])
        for sr in out["shape_residuals"]:
            assert set(sr) == {"l2_norm", "membrane_energy", "bending_energy"}
            assert all(np.isfinite(v) and v >= 0 for v in sr.values())

    def test_per_image_outputs(self, syn_result):
        out, _ = syn_result
        assert len(out["warped_images"]) == 4
        for fwd, inv in zip(out["fwdtransforms"], out["invtransforms"]):
            # ants.registration() convention: [warp, affine] / [affine, inverse warp]
            assert len(fwd) == 2 and len(inv) == 2
            assert fwd[0].endswith("1Warp.nii.gz") and fwd[1].endswith("0GenericAffine.mat")
            assert inv[0].endswith("0GenericAffine.mat") and inv[1].endswith("1InverseWarp.nii.gz")
            assert all(os.path.exists(p) for p in fwd + inv)

    def test_work_dir_artifacts(self, syn_result):
        out, work = syn_result
        assert out["work_dir"] == str(work)
        assert (work / "avgAffine_0.mat").exists()
        assert (work / "avgWarp_0.nii.gz").exists()
        assert (work / "avgWarp_1.nii.gz").exists()

    def test_second_iteration_reuses_affine(self, syn_result):
        # SyNOnly from the cached affine: the affine file written at iteration 1
        # must equal the one fitted at iteration 0 (no affine drift).
        _, work = syn_result
        a0 = ants.read_transform(str(work / "iter00_img0000" / "out0GenericAffine.mat"))
        a1 = ants.read_transform(str(work / "iter01_img0000" / "out0GenericAffine.mat"))
        np.testing.assert_allclose(a0.parameters, a1.parameters, atol=1e-4)

    def test_weights_normalised(self, syn_result):
        out, _ = syn_result
        np.testing.assert_allclose(out["weights"], [0.25] * 4)

    def test_default_work_dir_is_temporary(self):
        out = build_template(image_list=_cohort(offsets=((-2, 0), (2, 0))), iterations=1, **FAST)
        assert os.path.isdir(out["work_dir"])
        assert "antstorch_tmpl_" in os.path.basename(out["work_dir"])


class TestOptions:
    def test_initial_template_is_used_and_not_mutated(self):
        imgs = _cohort(offsets=((-2, 0), (2, 0)))
        start = _disc(16, 16)
        before = start.numpy().copy()
        out = build_template(initial_template=start, image_list=imgs, iterations=1, **FAST)
        np.testing.assert_array_equal(start.numpy(), before)
        assert out["template"].shape == start.shape

    def test_no_blending_matches_weight_one(self):
        imgs = _cohort(offsets=((-2, 0), (2, 0)))
        out = build_template(image_list=imgs, iterations=1, blending_weight=1.0, **FAST)
        assert np.isfinite(out["template"].numpy()).all()

    def test_convergence_threshold_stops_early(self):
        out = build_template(
            image_list=_cohort(offsets=((-2, 0), (2, 0))),
            iterations=4,
            convergence_threshold=1e6,
            **FAST,
        )
        assert out["n_iterations"] == 1

    def test_affine_every_iteration(self, tmp_path):
        build_template(
            image_list=_cohort(offsets=((-2, 0), (2, 0))),
            iterations=2,
            affine_every_iteration=True,
            output_dir=str(tmp_path),
            **FAST,
        )
        # full registration at iteration 1 as well: affine fitted again, warp still present
        assert (tmp_path / "iter01_img0000" / "out0GenericAffine.mat").exists()
        assert (tmp_path / "iter01_img0000" / "out1Warp.nii.gz").exists()

    def test_use_no_rigid_false(self):
        out = build_template(
            image_list=_cohort(offsets=((-2, 0), (2, 0))), iterations=1, use_no_rigid=False, **FAST
        )
        assert np.isfinite(out["template"].numpy()).all()

    def test_linear_only_type_has_no_warp(self):
        out = build_template(
            image_list=_cohort(offsets=((-2, 0), (2, 0))),
            iterations=1,
            type_of_transform="Rigid",
        )
        assert out["shape_residuals"][0] == {"l2_norm": 0.0, "membrane_energy": 0.0, "bending_energy": 0.0}
        assert all(len(f) == 1 for f in out["fwdtransforms"])

    def test_syn_kwargs_forwarded(self):
        # a regularizer keyword must reach syn_registration without error
        out = build_template(
            image_list=_cohort(offsets=((-2, 0), (2, 0))), iterations=1, regularizer="sobolev", **FAST
        )
        assert np.isfinite(out["template"].numpy()).all()


class TestAffineNotReapplied:
    """The cached iteration-0 affine must not be averaged into the template again at later iterations."""

    def test_mean_affine_only_at_iteration_zero(self, tmp_path):
        build_template(
            image_list=_cohort(offsets=((-2, 0), (2, 0))), iterations=3, output_dir=str(tmp_path), **FAST
        )
        assert (tmp_path / "avgAffine_0.mat").exists()
        assert not (tmp_path / "avgAffine_1.mat").exists()
        assert not (tmp_path / "avgAffine_2.mat").exists()
        # the dense warp still recentres the template at every iteration
        assert all((tmp_path / f"avgWarp_{i}.nii.gz").exists() for i in range(3))

    def test_affine_every_iteration_averages_each_time(self, tmp_path):
        build_template(
            image_list=_cohort(offsets=((-2, 0), (2, 0))),
            iterations=2,
            affine_every_iteration=True,
            output_dir=str(tmp_path),
            **FAST,
        )
        assert (tmp_path / "avgAffine_0.mat").exists()
        assert (tmp_path / "avgAffine_1.mat").exists()


def test_identical_images_are_a_fixed_point():
    """Registering identical images leaves the template (almost) unchanged and the mean warp tiny."""
    yy, xx = np.mgrid[0:48, 0:48].astype(np.float64)
    blob = ants.from_numpy(np.exp(-(((xx - 24) / 7.0) ** 2 + ((yy - 24) / 7.0) ** 2)).astype(np.float32))
    out = build_template(
        image_list=[blob, blob.clone(), blob.clone()], iterations=2, blending_weight=1.0, **FAST
    )
    corr = np.corrcoef(out["template"].numpy().ravel(), blob.numpy().ravel())[0, 1]
    assert corr > 0.99
    assert max(sr["l2_norm"] for sr in out["shape_residuals"]) < 0.5


# ---------------------------------------------------------------------------
# NumPy fallback for antsApplyTransforms(imagetype=1) on vector images
# ---------------------------------------------------------------------------


def _vector_case(dim, invert, tmp_path):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(7)
    shape = (24, 22) if dim == 2 else (14, 12, 10)
    spacing = list(rng.uniform(0.8, 1.5, dim))
    kw = dict(spacing=spacing, origin=[0.5] * dim)
    fixed = ants.from_numpy(rng.random(shape).astype("float32"), **kw)
    data = rng.normal(size=shape + (dim,)).astype("float32")
    for k in range(dim):
        data[..., k] = gaussian_filter(data[..., k], 2)
    moving = ants.from_numpy(data, has_components=True, **kw)
    matrix = np.eye(dim) + rng.uniform(-0.12, 0.12, (dim, dim))
    tx = ants.create_ants_transform(
        transform_type="AffineTransform", dimension=dim, matrix=matrix, translation=rng.uniform(-1, 1, dim),
        center=[3.0] * dim,
    )
    fn = str(tmp_path / f"affine_{dim}d.mat")
    ants.write_transform(tx, fn)
    return fixed, moving, fn


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("invert", [True, False])
def test_vector_fallback_matches_ants(dim, invert, tmp_path):
    from antstorch.syn.template import _apply_affine_to_vector_image

    fixed, moving, fn = _vector_case(dim, invert, tmp_path)
    try:
        ref = ants.apply_transforms(
            fixed=fixed, moving=moving, imagetype=1, transformlist=[fn], whichtoinvert=[1 if invert else 0]
        )
    except RuntimeError:
        pytest.skip("this ANTsPy build cannot apply a transform to a vector image (the fallback exists for that case)")
    got = _apply_affine_to_vector_image(fixed, moving, fn, invert=invert)
    assert got.components == dim
    inner = tuple(slice(3, -3) for _ in range(dim))
    np.testing.assert_allclose(got.numpy()[inner], ref.numpy()[inner], atol=1e-5)


def test_vector_fallback_runs_without_ants_vector_support(tmp_path, monkeypatch):
    """With ANTs' vector path unavailable, ``_apply_affine_to_vectors`` returns the NumPy result."""
    from antstorch.syn import template as tmod

    fixed, moving, fn = _vector_case(2, True, tmp_path)
    real = ants.apply_transforms

    def fake(fixed, moving, **kwargs):
        if kwargs.get("imagetype", 0) != 0 or moving.components > 1:
            raise RuntimeError("antsApplyTransforms returned non-zero exit code 1")
        return real(fixed=fixed, moving=moving, **kwargs)

    monkeypatch.setattr(tmod.ants, "apply_transforms", fake)
    out = tmod._apply_affine_to_vectors(fixed, moving, fn, invert=True)
    assert out.components == 2 and out.shape == moving.shape
    assert np.isfinite(out.numpy()).all()


# ---------------------------------------------------------------------------
# torch template update (multi-component images, non-linear interpolators)
# ---------------------------------------------------------------------------

_DIRECTION = np.diag([-1.0, -1.0, 1.0])
FAST_3D = dict(levels=(2, 1), reg_iterations=(3, 2), affine_iterations=(10, 10, 10), affine_multi_start=False)


def _blob_3d(shape, center, sigma=3.0, spacing=(2.0, 2.0, 2.0), origin=(-20.0, -18.0, 5.0)):
    """A scalar Gaussian blob plus a smaller satellite (asymmetric, so the affine cannot land on a reflection)."""
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float64)

    def gaussian(c, s):
        return np.exp(-(((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / (2.0 * s**2)))

    satellite = tuple(c + o for c, o in zip(center, (2.5, -2.0, 1.5)))
    array = gaussian(center, sigma) + 0.6 * gaussian(satellite, sigma / 2.0)
    array = np.ascontiguousarray(np.transpose(array, (2, 1, 0)).astype(np.float32))
    return ants.from_numpy(array, origin=origin, spacing=spacing, direction=_DIRECTION)


def _two_channel(image):
    array = image.numpy()
    return ants.from_numpy(
        np.ascontiguousarray(np.stack([array, 0.5 * array], axis=-1)), origin=image.origin, spacing=image.spacing,
        direction=image.direction, has_components=True,
    )


def _smooth_warp(reference, amplitude=1.5):
    """A smooth displacement field (vector ANTsImage on ``reference``'s grid, ITK (x, y, z) components)."""
    x, y, z = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in reference.shape], indexing="ij")
    field = np.stack(
        [amplitude * np.sin(x / 4.0) * np.cos(y / 5.0), -amplitude * np.cos(z / 3.0), amplitude * np.sin((x + y) / 6.0)],
        axis=-1,
    ).astype(np.float32)
    return ants.from_numpy(
        field, origin=reference.origin, spacing=reference.spacing, direction=reference.direction, has_components=True
    )


class TestTorchTemplateUpdate:
    def test_matches_ants_apply_transforms(self, tmp_path):
        rng = np.random.default_rng(0)
        scalar = _blob_3d((14, 15, 16), (7, 8, 7)) + ants.from_numpy(
            rng.random((16, 15, 14), dtype=np.float32) * 0.1, origin=(-20.0, -18.0, 5.0), spacing=(2.0, 2.0, 2.0),
            direction=_DIRECTION,
        )
        # ANTsPy has no 1-component vector image; duplicate the scalar into two channels.
        vector = _two_channel(scalar)
        warp = _smooth_warp(scalar)
        warp_fn = str(tmp_path / "warp.nii.gz")
        ants.image_write(warp, warp_fn)
        angle = np.radians(8.0)
        matrix = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.05]])
        affine_fn = str(tmp_path / "affine.mat")
        write_affine_transform(torch.as_tensor(matrix), torch.as_tensor([1.0, -2.0, 0.5]), 3, affine_fn)
        inner = (slice(2, -2),) * 3
        for transforms, invert, w, a in (
            ([warp_fn, affine_fn], [0, 1], warp, affine_fn), ([warp_fn], [0], warp, None), ([affine_fn], [1], None, affine_fn),
        ):
            expected = ants.apply_transforms(fixed=scalar, moving=scalar, transformlist=transforms, whichtoinvert=invert)
            got = _apply_transforms_torch(vector, w, a)
            assert got.components == 2
            np.testing.assert_allclose(got.numpy()[..., 0][inner], expected.numpy()[inner], atol=2e-3)
            np.testing.assert_allclose(got.numpy()[..., 1][inner], 0.5 * expected.numpy()[inner], atol=2e-3)

    def test_identity_and_one_voxel_shift(self, tmp_path):
        rng = np.random.default_rng(1)
        scalar = ants.from_numpy(
            np.ascontiguousarray(rng.random((12, 11, 10), dtype=np.float32)), origin=(-20.0, -18.0, 5.0),
            spacing=(2.0, 2.0, 2.0), direction=_DIRECTION,
        )
        identity_fn = str(tmp_path / "identity.mat")
        write_affine_transform(torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 3, identity_fn)
        zero_warp = _smooth_warp(scalar, amplitude=0.0)
        for interpolator in ("linear", "cubic", "nearest"):
            got = _apply_transforms_torch(scalar, zero_warp, identity_fn, interpolator)
            np.testing.assert_allclose(got.numpy(), scalar.numpy(), atol=1e-5)
        # the approximating B-spline does not reproduce the voxel values
        bspline = _apply_transforms_torch(scalar, zero_warp, identity_fn, "bspline").numpy()
        assert np.abs(bspline - scalar.numpy()).max() > 1e-2
        # a translation by one voxel is exact for an interpolating kernel (away from the borders)
        shift_fn = str(tmp_path / "shift.mat")
        write_affine_transform(torch.eye(3, dtype=torch.float64), torch.tensor([-2.0, 0.0, 0.0]), 3, shift_fn)
        got = _apply_transforms_torch(scalar, None, shift_fn, "cubic").numpy()
        np.testing.assert_allclose(got[1:, :, :], scalar.numpy()[:-1, :, :], atol=1e-5)

    def test_sharpen_image_is_per_channel(self):
        image = _two_channel(_blob_3d((10, 11, 12), (5, 5, 6)))
        out = _sharpen_image(image, 0.75)
        assert out.components == 2 and tuple(out.shape) == tuple(image.shape)
        channel = ants.split_channels(image)[0]
        expected = channel * 0.75 + ants.iMath(channel, "Sharpen") * 0.25
        np.testing.assert_allclose(out.numpy()[..., 0], expected.numpy(), atol=1e-5)
        np.testing.assert_allclose(np.asarray(out.direction), np.asarray(image.direction))

    @pytest.mark.parametrize("interpolator", ["linear", "cubic"])
    def test_resample_vector_onto_identity_and_shifted_grid(self, interpolator):
        image = _two_channel(_blob_3d((10, 11, 12), (5, 5, 6)))
        np.testing.assert_allclose(_resample_vector_onto(image, image, interpolator=interpolator).numpy(), image.numpy())
        # a copy of the grid (not the same object) still reproduces the voxel values with an interpolating kernel
        copy = image * 0.0
        np.testing.assert_allclose(
            _resample_vector_onto(image, copy, interpolator=interpolator).numpy(), image.numpy(), atol=1e-5
        )
        shifted = _two_channel(_blob_3d((10, 11, 12), (5, 5, 6), origin=(-19.0, -17.0, 6.0)))
        out = _resample_vector_onto(image, shifted, interpolator=interpolator)
        inner = (slice(1, -1),) * 3
        if interpolator == "cubic":
            # The target grid's origin is 1 mm further along each LPS axis, half a voxel: the blob
            # (centered in (z, y, x) index order) moves to (cz - 0.5, cy + 0.5, cx + 0.5).
            truth = _two_channel(_blob_3d((10, 11, 12), (4.5, 5.5, 6.5))).numpy()
            linear = _resample_vector_onto(image, shifted, interpolator="linear").numpy()
            assert np.abs(out.numpy() - truth)[inner].mean() < np.abs(linear - truth)[inner].mean()
            return
        expected = ants.resample_image_to_target(
            ants.split_channels(image)[0], ants.split_channels(shifted)[0], interp_type="linear"
        ).numpy()
        np.testing.assert_allclose(out.numpy()[..., 0][inner], expected[inner], atol=1e-4)

    def test_cubic_scalar_template(self, tmp_path):
        images = [_blob_3d((12, 13, 14), c) for c in ((6, 6, 7), (5, 7, 8), (7, 6, 6))]
        result = build_template(
            image_list=images, iterations=1, blending_weight=1.0, output_dir=str(tmp_path), output_interpolator="cubic",
            **FAST_3D,
        )
        assert result["template"].components == 1 and np.isfinite(result["template"].numpy()).all()
        assert tuple(result["template"].shape) == tuple(images[0].shape)
        with pytest.raises(ValueError, match="output_interpolator"):
            build_template(image_list=images, iterations=1, output_interpolator="sinc", **FAST_3D)

    def test_multicomponent_template_with_sharpening(self, tmp_path):
        images = [_two_channel(_blob_3d((12, 13, 14), c)) for c in ((6, 6, 7), (5, 7, 8), (7, 6, 6))]
        result = build_template(image_list=images, iterations=1, blending_weight=0.75, output_dir=str(tmp_path), **FAST_3D)
        assert result["template"].components == 2 and np.isfinite(result["template"].numpy()).all()
        with pytest.raises(ValueError, match="same number of components"):
            build_template(image_list=[images[0], _blob_3d((12, 13, 14), (6, 6, 7))], iterations=1, **FAST_3D)


# ---------------------------------------------------------------------------
# extra contrasts, continued builds, linear pre-alignment
# ---------------------------------------------------------------------------

_CENTERS = ((6, 6, 7), (5, 7, 8), (7, 6, 6))


class TestMultiContrast:
    def test_extra_contrast(self, tmp_path):
        images = [_blob_3d((12, 13, 14), c) for c in _CENTERS]
        extras = [_blob_3d((12, 13, 14), c, sigma=2.5) * 3.0 for c in _CENTERS]
        common = dict(output_dir=str(tmp_path), **FAST_3D)
        result = build_template(
            image_list=images, extra_contrasts=[extras], extra_contrast_weights=[0.5], iterations=2,
            blending_weight=0.75, **common,
        )
        assert len(result["extra_templates"]) == 1
        extra_template = result["extra_templates"][0]
        assert extra_template.components == 1 and tuple(extra_template.shape) == tuple(images[0].shape)
        assert np.allclose(extra_template.spacing, images[0].spacing) and np.isfinite(extra_template.numpy()).all()
        # The extra template is an average of the warped extras: its peak sits between the inputs' peaks.
        assert 0.5 * 3.0 < extra_template.numpy().max() <= 3.0 * 1.6 * 1.05
        assert len(result["warped_extra_images"]) == 1 and len(result["warped_extra_images"][0]) == 3
        assert result["n_iterations"] == 2
        # Extras on a finer grid are resampled onto the primary grid first.
        fine = [_blob_3d((24, 26, 28), tuple(2 * v for v in c), sigma=5.0, spacing=(1.0, 1.0, 1.0)) for c in _CENTERS]
        fine_result = build_template(image_list=images, extra_contrasts=[fine], iterations=1, blending_weight=1.0, **common)
        assert tuple(fine_result["extra_templates"][0].shape) == tuple(images[0].shape)
        assert np.allclose(fine_result["extra_templates"][0].spacing, images[0].spacing)
        with pytest.raises(ValueError, match="extra_contrasts\\[0\\]"):
            build_template(image_list=images, extra_contrasts=[extras[:2]], iterations=1, **common)
        with pytest.raises(ValueError, match="extra_contrast_weights"):
            build_template(
                image_list=images, extra_contrasts=[extras], extra_contrast_weights=[1.0, 1.0], iterations=1, **common
            )
        with pytest.raises(ValueError, match="scalar"):
            build_template(image_list=images, extra_contrasts=[[_two_channel(e) for e in extras]], iterations=1, **common)

    def test_without_extra_contrasts_returns_empty_lists(self, tmp_path):
        images = [_blob_3d((10, 10, 10), c) for c in ((5, 5, 5), (4, 5, 6))]
        result = build_template(
            image_list=images, iterations=1, output_dir=str(tmp_path), type_of_transform="SyNOnly", levels=(1,),
            reg_iterations=(1,),
        )
        assert result["extra_templates"] == [] and result["warped_extra_images"] == []

    def test_continues_from_previous_templates(self, tmp_path):
        images = [_blob_3d((12, 13, 14), c) for c in _CENTERS]
        extras = [_blob_3d((12, 13, 14), c, sigma=2.5) * 3.0 for c in _CENTERS]
        common = dict(extra_contrasts=[extras], iterations=1, blending_weight=1.0, **FAST_3D)
        first = build_template(image_list=images, output_dir=str(tmp_path / "first"), **common)
        second = build_template(
            image_list=images, output_dir=str(tmp_path / "second"), initial_template=first["template"],
            initial_extra_templates=first["extra_templates"], **common,
        )
        extra_template = second["extra_templates"][0]
        assert second["n_iterations"] == 1 and extra_template.components == 1
        assert np.isfinite(extra_template.numpy()).all()
        # One more iteration refines the extra template rather than restarting from scratch.
        assert np.abs(extra_template.numpy() - first["extra_templates"][0].numpy()).max() < 0.5 * 3.0
        with pytest.raises(ValueError, match="initial_extra_templates must hold one image per extra contrast"):
            build_template(
                image_list=images, output_dir=str(tmp_path / "bad"), initial_template=first["template"],
                initial_extra_templates=[], **common,
            )

    def test_initial_alignment_starts_from_aligned_inputs(self, tmp_path):
        centers = ((6, 6, 7), (4, 7, 8), (8, 6, 6))
        images = [_blob_3d((12, 13, 14), c) for c in centers]
        extras = [_blob_3d((12, 13, 14), c, sigma=2.5) * 3.0 for c in centers]
        common = dict(blending_weight=1.0, iterations=1, **dict(FAST_3D, reg_iterations=(2, 2)))
        res = build_template(
            image_list=images, extra_contrasts=[extras], output_dir=str(tmp_path / "aligned"),
            initial_alignment="Rigid", initial_alignment_iterations=2, **common,
        )
        pre = res["initial_alignment"]
        assert pre is not None and pre["reference"] == "image 0" and res["n_iterations"] == 1
        # Pass 1 targets image 0, pass 2 the running average.
        assert [r["n_iterations"] for r in pre["passes"]] == [1, 1]
        for name in ("initial_rigid_reference", "initial_rigid"):
            assert os.path.isfile(os.path.join(tmp_path, "aligned", name, "iter00_img0000", "out0GenericAffine.mat"))
        assert np.isfinite(res["template"].numpy()).all() and np.isfinite(res["extra_templates"][0].numpy()).all()
        # Rigidly aligning the blobs before averaging gives a sharper starting template than the plain average.
        plain = np.mean([e.numpy() for e in extras], axis=0)
        assert pre["extra_templates"][0].numpy().max() > plain.max() + 0.1 * 3.0
        # A supplied initial template skips the pre-alignment.
        again = build_template(
            image_list=images, extra_contrasts=[extras], output_dir=str(tmp_path / "given"), initial_alignment="Rigid",
            initial_template=res["template"], initial_extra_templates=res["extra_templates"], **common,
        )
        assert again["initial_alignment"] is None
        with pytest.raises(ValueError, match="initial_alignment must be one of"):
            build_template(image_list=images, initial_alignment="SyN", **common)
        with pytest.raises(ValueError, match="not an index into image_list"):
            build_template(image_list=images, initial_alignment="Rigid", initial_alignment_reference=3, **common)
        with pytest.raises(ValueError, match="must be a sequence"):
            build_template(
                image_list=images, extra_contrasts=[extras], initial_alignment="Rigid",
                initial_alignment_reference=images[1], **common,
            )

    def test_initial_alignment_reference_image_breaks_ghosts_in_the_average(self, tmp_path):
        # Two subsets of inputs that already share a pose, 8 voxels apart: the plain average shows two ghosts,
        # and a first pass against that average leaves the inputs split between them; a first pass against one
        # input gathers everything.
        def blob(center, shape=(16, 12, 12), sigma=1.5):
            grid = np.indices(shape, dtype=np.float32)
            array = np.exp(-sum((g - c) ** 2 for g, c in zip(grid, center)) / (2.0 * sigma**2))
            return ants.from_numpy(np.ascontiguousarray(array))

        images = [blob(c) for c in ((4, 6, 6), (4, 6, 6), (12, 6, 6), (12, 6, 6))]
        common = dict(
            type_of_transform="Translation", initial_alignment="Translation", affine_iterations=(40, 40, 40),
            affine_multi_start=False, iterations=1, blending_weight=1.0,
        )
        gathered = build_template(image_list=images, output_dir=str(tmp_path / "ref"), initial_alignment_reference=0, **common)
        # Against the ghosted average each input starts on top of its own ghost and stays there (a
        # center-of-mass start would drop identical blobs onto the saddle between the ghosts instead).
        ghosted = build_template(
            image_list=images, output_dir=str(tmp_path / "avg"), initial_alignment_reference=None,
            affine_center_of_mass_init=False, **common,
        )
        assert gathered["initial_alignment"]["reference"] == "image 0"
        assert ghosted["initial_alignment"]["reference"] == "the average of the unaligned inputs"
        assert gathered["initial_alignment"]["template"].numpy().max() > 0.85
        assert ghosted["initial_alignment"]["template"].numpy().max() < 0.7
        # The gathered template sits at the group's mean pose (between the two subsets), not on the reference.
        peak = np.unravel_index(gathered["initial_alignment"]["template"].numpy().argmax(), (16, 12, 12))
        assert 6 <= peak[0] <= 10
