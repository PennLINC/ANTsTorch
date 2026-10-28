"""Fiber orientation distribution (FOD) registration and templates for :mod:`antstorch.syn`.

:func:`fod_registration` and :func:`build_fod_template` run
:func:`antstorch.syn.syn_registration` and :func:`antstorch.syn.build_template`
with a :class:`FODReorientation`.

An FOD image is a multi-component image whose channels are the coefficients of
an even-degree real spherical harmonic (SH) series in the MRtrix3 convention
(see :mod:`antstorch.syn.sh`). Registering such images needs three things on
top of a multichannel SyN:

* **reorientation**: when a voxel is moved by a spatially varying map, the
  FOD inside it must be rotated and sheared by the local Jacobian so that the
  fibers keep pointing along the (deformed) tracts. This module supplies the
  per-voxel Jacobians of the SyN half-warps, in the frame the SH series is
  expressed in, and applies the apodized-PSF reorientation of
  :class:`antstorch.syn.sh.FODReorienter` to the images sampled at the SyN
  midpoint and to the final warped outputs;
* **a degree schedule**: like ``mrregister``, coarse pyramid levels are
  registered on a truncated series (lmax 0, then 2, then 4 by default);
* **frames**: ITK/ANTs physical coordinates are LPS while MRtrix3 SH images
  are expressed in RAS scanner coordinates, so the Jacobians are conjugated by
  ``diag(-1, -1, 1)`` before they act on fiber directions, matching
  ``mrtransform -reorient_fod yes``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import os

import nibabel as nib
import numpy as np
import torch
from torch import Tensor

try:
    import odx
except ImportError:
    odx = None

from .core.jacobian import _spatial_jacobian_nd
from .sh import FODReorienter, lmax_from_n, n_coefficients
from .syn import syn_registration
from .template import build_template

__all__ = [
    "sh_lmax_of_image",
    "fod_image_read",
    "fod_image_write",
    "default_fod_lmax",
    "pullback_jacobian",
    "direction_transform",
    "FODReorientation",
    "fod_registration",
    "build_fod_template",
]

_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def sh_lmax_of_image(image) -> int:
    """lmax of a multi-component ``ants.ANTsImage`` holding an even-degree SH series."""
    components = int(image.components)
    if components < 2:
        raise ValueError("an FOD image must have more than one component (SH coefficient)")
    return lmax_from_n(components)


_ODX_READERS = {
    ".odx": "load", ".mif": "from_mrtrix", ".mif.gz": "from_mrtrix", ".fib.gz": "from_fibgz", ".fz": "from_fz",
}
_ODX_WRITERS = (".odx", ".mif.gz")


def _extension(filename: str) -> str:
    name = str(filename).lower()
    for extension in (".nii.gz", ".mif.gz", ".fib.gz"):
        if name.endswith(extension):
            return extension
    return os.path.splitext(name)[1]


def _require_odx(filename: str):
    if odx is None:
        raise ImportError(f"{filename}: reading and writing this format needs the odx package (pip install odx)")
    return odx


def _ras_affine(image) -> np.ndarray:
    """The voxel-to-RAS affine of a 3-D ``ants.ANTsImage`` (NIfTI and MRtrix3 convention)."""
    direction = np.asarray(image.direction, dtype=np.float64).reshape(3, 3)
    affine_lps = np.eye(4)
    affine_lps[:3, :3] = direction * np.asarray(image.spacing, dtype=np.float64)[None, :]
    affine_lps[:3, 3] = np.asarray(image.origin, dtype=np.float64)
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps


def _from_ras_affine(array: np.ndarray, affine_ras: np.ndarray):
    """A multi-component ``ants.ANTsImage`` from a channel-last ``(x, y, z, n)`` array and its voxel-to-RAS affine."""
    import ants

    affine_lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(affine_ras, dtype=np.float64)
    spacing = np.linalg.norm(affine_lps[:3, :3], axis=0)
    return ants.from_numpy(
        np.ascontiguousarray(array, dtype=np.float32),
        origin=tuple(float(v) for v in affine_lps[:3, 3]),
        spacing=tuple(float(v) for v in spacing),
        direction=affine_lps[:3, :3] / spacing[None, :],
        has_components=True,
    )


def fod_image_read(filename: str):
    """Read an SH FOD image as a 3-D multi-component ``ants.ANTsImage`` in the MRtrix3 SH convention.

    4-D NIfTI files (as written by MRtrix3) are read with ANTsPy: the fourth
    axis becomes the components and the coefficients are kept exactly as
    stored (MRtrix3 expresses them in the RAS scanner frame); only the
    geometry is converted to ITK's LPS metadata. ODX (``.odx``), MRtrix3
    (``.mif``, ``.mif.gz``) and DSI Studio (``.fib.gz``, ``.fz``) files are read
    with the optional ``odx`` package, their SH coefficients converted to the
    MRtrix3 basis (``tournier07``). The result can be passed to
    :func:`fod_registration` or :func:`build_fod_template`.
    """
    import ants

    extension = _extension(filename)
    if extension in _ODX_READERS:
        reader = getattr(_require_odx(filename), _ODX_READERS[extension])
        dataset = reader(str(filename))
        if not dataset.sh_names:
            raise ValueError(f"{filename}: the dataset holds no SH coefficients")
        if dataset.sh_full_basis:
            raise ValueError(f"{filename}: full-basis (asymmetric) SH series are not supported")
        if dataset.sh_basis != "tournier07":
            dataset = dataset.convert_sh_basis_to("tournier07")
        return _from_ras_affine(dataset.densify_sh(), dataset.affine)
    image = ants.image_read(filename)
    if image.dimension == 3 and image.components > 1:
        return image
    if image.dimension != 4:
        raise ValueError(f"{filename}: expected a 4-D image of SH coefficients, got a {image.dimension}-D image")
    array = np.ascontiguousarray(image.numpy().astype(np.float32, copy=False))  # (x, y, z, n)
    direction = np.asarray(image.direction, dtype=np.float64).reshape(4, 4)[:3, :3]
    return ants.from_numpy(
        array,
        origin=tuple(float(v) for v in image.origin[:3]),
        spacing=tuple(float(v) for v in image.spacing[:3]),
        direction=direction,
        has_components=True,
    )


def fod_image_write(image, filename: str) -> None:
    """Write a multi-component ``ants.ANTsImage`` of SH coefficients (MRtrix3 convention).

    ``.nii``/``.nii.gz`` give a 4-D NIfTI with a RAS affine that MRtrix3 reads
    as an SH image; ``.odx`` and ``.mif.gz`` are written with the optional
    ``odx`` package (voxels whose coefficients are all zero are left out of
    the ODX mask).
    """
    if image.dimension != 3:
        raise ValueError("fod_image_write expects a 3-D multi-component image")
    array = np.asarray(image.numpy(), dtype=np.float32)
    if array.ndim == 3:
        array = array[..., None]
    affine_ras = _ras_affine(image)
    extension = _extension(filename)
    if extension in _ODX_WRITERS:
        mask = np.ascontiguousarray(np.any(array != 0, axis=-1), dtype=np.uint8)
        dataset = _require_odx(filename).from_sh_coefficients(
            np.ascontiguousarray(array), mask=mask, affine=affine_ras, basis="tournier07",
            sh_order=lmax_from_n(array.shape[-1]), legacy=False, compute_peaks=False,
        )
        if extension == ".odx":
            odx.save(str(filename), dataset)
        else:
            dataset.save_mif(str(filename), which="sh")
        return
    if extension == ".mif":
        raise ValueError(f"{filename}: write MRtrix3 images as .mif.gz")
    nib.save(nib.Nifti1Image(np.ascontiguousarray(array), affine_ras), filename)


def default_fod_lmax(num_levels: int, image_lmax: int, max_lmax: int = 4) -> Tuple[int, ...]:
    """MRtrix3-like degree schedule: lmax 0 at the coarsest level, then 2, 4, ....

    The values are capped at ``min(max_lmax, image_lmax)``.

    ``mrregister`` registers FODs with lmax 0, 2 and 4 at scale factors 0.25,
    0.5 and 1.0; this reproduces that for three levels and keeps the cap for
    longer pyramids.
    """
    cap = min(int(max_lmax), int(image_lmax))
    cap -= cap % 2
    return tuple(min(2 * i, cap) for i in range(num_levels))


def _inverse_3x3(matrix: Tensor, eps: float = 1e-6) -> Tensor:
    """Batched 3x3 inverse by adjugate, with the determinant kept away from zero.

    Folded voxels have singular Jacobians, where ``torch.linalg.inv`` raises or returns inf;
    the clamped determinant keeps their reorientation finite.
    """
    a, b, c = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    d, e, f = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    g, h, i = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    c00 = e * i - f * h
    c01 = -(d * i - f * g)
    c02 = d * h - e * g
    c10 = -(b * i - c * h)
    c11 = a * i - c * g
    c12 = -(a * h - b * g)
    c20 = b * f - c * e
    c21 = -(a * f - c * d)
    c22 = a * e - b * d
    det = a * c00 + b * c01 + c * c02
    det = torch.where(det.abs() < eps, torch.where(det < 0, -torch.full_like(det, eps), torch.full_like(det, eps)), det)
    adjugate = torch.stack(
        [
            torch.stack([c00, c10, c20], dim=-1),
            torch.stack([c01, c11, c21], dim=-1),
            torch.stack([c02, c12, c22], dim=-1),
        ],
        dim=-2,
    )
    return adjugate / det[..., None, None]


def pullback_jacobian(
    displacement: Tensor, spacing: Sequence[float], direction, affine: Optional[Tensor] = None
) -> Tensor:
    """Jacobian of a pull-back map ``phi(x) = A (x + u(x)) + b`` in ITK physical ``(x, y, z)`` coordinates.

    Parameters
    ----------
    displacement : Tensor
        ``u``, shape ``(B, *spatial, 3)`` in physical mm with components in the
        reversed ``(z, y, x)`` order used throughout :mod:`antstorch.syn`
        (a SyN half-warp or a total forward/inverse field).
    spacing : sequence of float
        Voxel spacing of the grid ``u`` lives on, ITK ``(x, y, z)`` order.
    direction : array-like
        ``(3, 3)`` ITK direction cosines of that grid.
    affine : Tensor, optional
        ``A`` as a ``(3, 3)`` matrix in the reversed ``(z, y, x)`` component
        order (``M_phys`` in :func:`antstorch.syn.syn_registration`), applied
        after the displacement; ``None`` for the identity.

    Returns
    -------
    Tensor
        ``(B, *spatial, 3, 3)`` matrices ``d phi / d x`` with ``(x, y, z)``
        row and column order. Central differences inside the volume,
        one-sided at the faces.
    """
    if displacement.shape[-1] != 3:
        raise ValueError("pullback_jacobian supports 3-D fields only")
    jac = _spatial_jacobian_nd(displacement, physical_spacing=tuple(float(s) for s in reversed(spacing)))
    jac = jac.flip(-1).flip(-2)  # (z, y, x) -> (x, y, z) for both the component and the derivative index
    cosines = torch.as_tensor(np.asarray(direction, dtype=np.float64).reshape(3, 3), device=jac.device, dtype=jac.dtype)
    jac = jac @ cosines.transpose(0, 1)  # derivatives along the image axes -> along the physical axes
    jac = torch.eye(3, device=jac.device, dtype=jac.dtype) + jac
    if affine is not None:
        matrix_xyz = affine.to(device=jac.device, dtype=jac.dtype).flip(0).flip(1)
        jac = matrix_xyz @ jac
    return jac


def direction_transform(jacobian: Tensor, sh_frame: str = "RAS") -> Tensor:
    """Matrices that carry fiber directions of the input FOD to the output FOD.

    For a pull-back Jacobian ``J = d phi / d x`` this is ``J^-1`` (MRtrix3's
    ``Registration::Transform::reorient``), conjugated into the frame the SH
    series is expressed in: ``diag(-1, -1, 1) J^-1 diag(-1, -1, 1)`` for
    MRtrix3's RAS convention when ``J`` is in ITK's LPS coordinates.
    """
    inverse = _inverse_3x3(jacobian)
    frame = sh_frame.upper()
    if frame == "RAS":
        flip = torch.as_tensor(_LPS_TO_RAS, device=inverse.device, dtype=inverse.dtype)
        return flip @ inverse @ flip
    if frame == "LPS":
        return inverse
    raise ValueError("sh_frame must be 'RAS' (MRtrix3 SH images) or 'LPS'")


class FODReorientation:
    """How FOD images transform under a registration: the ``reorientation`` of :func:`antstorch.syn.syn_registration`.

    The channels are an even-degree SH series in the MRtrix3 convention. Each
    pyramid level registers the series truncated to :meth:`level_lmax`, the
    images sampled at the SyN midpoint are reoriented by the Jacobians of the
    half-warps, and the warped outputs by the Jacobians of the total maps,
    with the apodized-PSF kernels of :class:`antstorch.syn.sh.FODReorienter`
    (which this class builds and caches per degree, device and dtype).

    Parameters
    ----------
    image_lmax : int
        lmax of the SH images.
    lmax : int or sequence of int, default 4
        An int gives ``mrregister``'s schedule, lmax 0 at the coarsest level,
        then 2, 4, ..., capped at ``min(lmax, image_lmax)``
        (see :func:`default_fod_lmax`); a sequence gives one lmax per level.
    directions : int, str or array, default 60
        aPSF directions (see :func:`antstorch.syn.sh.load_directions`).
    modulate : bool, default False
        Apply FOD modulation (MRtrix3's ``-modulate fod``) to the warped
        output images.
    chunk_size : int
        Voxels per reorientation batch.
    sh_frame : str, default 'RAS'
        Frame of the SH basis, ``'RAS'`` for MRtrix3 images.
    """

    def __init__(
        self,
        image_lmax: int,
        *,
        lmax: Union[int, Sequence[int]] = 4,
        directions=60,
        modulate: bool = False,
        chunk_size: int = 1 << 14,
        sh_frame: str = "RAS",
    ):
        self.image_lmax = int(image_lmax)
        if isinstance(lmax, (int, np.integer)):
            if lmax < 0:
                raise ValueError(f"lmax must be non-negative, got {lmax}")
            self.lmax: Union[int, Tuple[int, ...]] = int(lmax)
        else:
            self.lmax = tuple(int(value) for value in lmax)
            for value in self.lmax:
                if value < 0 or value % 2 or value > self.image_lmax:
                    raise ValueError(
                        f"per-level lmax values must be even and in [0, {self.image_lmax}] "
                        f"(the images' lmax), got {self.lmax}"
                    )
        self.directions = directions
        self.modulate = bool(modulate)
        self.chunk_size = int(chunk_size)
        self.sh_frame = sh_frame
        self._reorienters: Dict[tuple, FODReorienter] = {}

    def __repr__(self) -> str:
        return (
            f"FODReorientation(image_lmax={self.image_lmax}, lmax={self.lmax}, "
            f"directions={self.directions!r}, modulate={self.modulate})"
        )

    def level_lmax(self, level_index: int) -> int:
        """lmax registered at pyramid level ``level_index`` (0 is the coarsest)."""
        if isinstance(self.lmax, int):
            return default_fod_lmax(level_index + 1, self.image_lmax, self.lmax)[-1]
        if level_index >= len(self.lmax):
            raise ValueError(f"lmax has {len(self.lmax)} entries; the registration has more levels")
        return self.lmax[level_index]

    def reorienter(self, lmax: int, device=None, dtype: torch.dtype = torch.float32) -> Optional[FODReorienter]:
        """The (cached) :class:`FODReorienter` for ``lmax``; ``None`` for lmax 0 (nothing to reorient)."""
        if lmax < 2:
            return None
        key = (int(lmax), str(device), dtype)
        if key not in self._reorienters:
            self._reorienters[key] = FODReorienter(lmax, self.directions, device=device, dtype=dtype)
        return self._reorienters[key]

    def channels(self, image: Tensor, level_index: int) -> Tensor:
        """The SH series up to the level's lmax."""
        return image[:, : n_coefficients(self.level_lmax(level_index))]

    def _reorient(self, image: Tensor, transform: Tensor, lmax: int, modulate: bool = False) -> Tensor:
        """Reorient the SH part of ``image`` (series up to ``lmax``); trailing channels pass through unchanged."""
        reorienter = self.reorienter(lmax, image.device, image.dtype)
        if reorienter is None:
            return image
        n = n_coefficients(lmax)
        sh_part, extras = image[:, :n], image[:, n:]
        out = reorienter.reorient(sh_part, transform, modulate=modulate, chunk_size=self.chunk_size)
        return out if extras.shape[1] == 0 else torch.cat([out, extras], dim=1)

    def midpoint_hook(self, level_index: int, fixed_meta: Dict[str, tuple], M_phys: Tensor) -> Optional[Callable]:
        """Callable reorienting the two images sampled at the SyN midpoint of a pyramid level.

        The returned function takes ``(I_mid, J_mid, warp_l2r, warp_r2l)`` and
        returns the reoriented ``(I_mid, J_mid)``. Both half-warps live on the
        level's fixed grid (``fixed_meta``); the moving image is additionally
        pulled back through the affine ``M_phys`` (reversed component order),
        so its Jacobian is ``M (I + d warp_r2l / dx)``. The Jacobians are
        computed from the detached warps: the gradient of the similarity flows
        through the sampled coefficients, not through the reorientation, as in
        ``mrregister``.
        """
        lmax = self.level_lmax(level_index)
        if lmax < 2:
            return None
        spacing, direction = fixed_meta["spacing"], fixed_meta["direction"]
        sh_frame = self.sh_frame

        def hook(I_mid: Tensor, J_mid: Tensor, warp_l2r: Tensor, warp_r2l: Tensor) -> Tuple[Tensor, Tensor]:
            with torch.no_grad():
                fixed_jacobian = pullback_jacobian(warp_l2r.detach(), spacing, direction)
                moving_jacobian = pullback_jacobian(warp_r2l.detach(), spacing, direction, affine=M_phys.detach())
                transform_fixed = direction_transform(fixed_jacobian, sh_frame)
                transform_moving = direction_transform(moving_jacobian, sh_frame)
            return self._reorient(I_mid, transform_fixed, lmax), self._reorient(J_mid, transform_moving, lmax)

        return hook

    def reorient_warped(
        self, image: Tensor, displacement: Tensor, meta: Dict[str, tuple], affine: Optional[Tensor] = None
    ) -> Tensor:
        """Reorient a full-lmax image sampled through ``A (x + displacement) + b`` on the grid ``meta`` describes.

        ``affine`` is ``A`` in the reversed ``(z, y, x)`` component order (see
        :func:`pullback_jacobian`); ``None`` for the identity.
        """
        with torch.no_grad():
            jacobian = pullback_jacobian(displacement, meta["spacing"], meta["direction"], affine=affine)
            transform = direction_transform(jacobian, self.sh_frame)
        return self._reorient(image, transform, self.image_lmax, self.modulate)

    def reorient_affine(self, image: Tensor, matrix_xyz: Tensor) -> Tensor:
        """Reorient a full-lmax image resampled through the pull-back affine ``x -> matrix_xyz x + t``.

        ``matrix_xyz`` is in ITK ``(x, y, z)`` order.
        """
        jacobian = matrix_xyz.detach().to(device=image.device, dtype=image.dtype)
        transform = direction_transform(jacobian, self.sh_frame)
        return self._reorient(image, transform, self.image_lmax, self.modulate)


def _check_fod_images(*images) -> int:
    """lmax shared by 3-D SH images with the same number of coefficients."""
    for image in images:
        if image.dimension != 3:
            raise ValueError(f"FOD images must be 3-D spherical harmonic images, got a {image.dimension}-D image")
    image_lmax = sh_lmax_of_image(images[0])
    if any(image.components != images[0].components for image in images):
        raise ValueError("all FOD images must have the same number of spherical harmonic coefficients")
    return image_lmax


def _check_level_count(lmax, kwargs) -> None:
    levels = kwargs.get("levels")
    if levels is not None and not isinstance(lmax, (int, np.integer)) and len(lmax) != len(levels):
        raise ValueError(f"lmax must have one entry per level ({len(levels)}), got {len(lmax)}")


def fod_registration(
    fixed,
    moving,
    *,
    lmax: Union[int, Sequence[int]] = 4,
    directions=60,
    modulate: bool = False,
    syn_metric: str = "lncc_joint",
    output_interpolator: str = "cubic",
    **kwargs: Any,
) -> Dict[str, object]:
    """Register two FOD images with :func:`antstorch.syn.syn_registration`, reorienting the FODs.

    ``fixed`` and ``moving`` are 3-D multi-component images whose channels are
    an even-degree SH series in the MRtrix3 convention (see
    :mod:`antstorch.syn.sh`; read MRtrix3 images with :func:`fod_image_read`).
    Both are scaled by one common factor (the 99th percentile of ``fixed``'s
    ``l = 0`` coefficient), the affine stage runs on the ``l = 0``
    coefficient, each pyramid level registers the series truncated to the
    level's lmax, the images sampled at the SyN midpoint are reoriented by the
    Jacobians of the half-warps with apodized point spread functions, and
    ``warpedmovout``/``warpedfixout`` are reoriented by the total maps.

    Parameters
    ----------
    fixed, moving : ants.ANTsImage
        SH images with the same number of coefficients.
    lmax : int or sequence of int, default 4
        Degree schedule (see :class:`FODReorientation`): an int caps
        ``mrregister``'s ``0, 2, 4, ...``; a sequence gives one lmax per level.
    directions : int, str or array, default 60
        aPSF directions for the reorientation (``60``, ``300`` or a direction file).
    modulate : bool, default False
        Apply FOD modulation (MRtrix3's ``-modulate fod``) to the warped outputs.
    syn_metric : str, default 'lncc_joint'
        Similarity over the coefficients; ``'mse'`` is the ``mrregister`` choice.
    output_interpolator : str, default 'cubic'
        Interpolation of the warped outputs; the interpolating cubic is the
        kernel of MRtrix3's ``mrtransform``.
    **kwargs
        Forwarded to :func:`antstorch.syn.syn_registration` (``type_of_transform``,
        ``levels``, ``reg_iterations``, ``extra_contrasts``, ``device``, ...).

    Returns
    -------
    dict
        :func:`antstorch.syn.syn_registration`'s result; its ``provenance``
        also records ``fod_lmax`` (per level), ``fod_directions`` and
        ``fod_modulate``.
    """
    image_lmax = _check_fod_images(fixed, moving)
    _check_level_count(lmax, kwargs)
    reorientation = FODReorientation(image_lmax, lmax=lmax, directions=directions, modulate=modulate)
    result = syn_registration(
        fixed, moving, reorientation=reorientation, syn_metric=syn_metric,
        output_interpolator=output_interpolator, **kwargs,
    )
    provenance = result["provenance"]
    levels = provenance.get("levels")
    provenance["fod_lmax"] = None if levels is None else tuple(reorientation.level_lmax(i) for i in range(len(levels)))
    provenance["fod_directions"] = directions
    provenance["fod_modulate"] = modulate
    return result


def build_fod_template(
    initial_template=None,
    image_list=None,
    *,
    lmax: Union[int, Sequence[int]] = 4,
    directions=60,
    modulate: bool = False,
    syn_metric: str = "lncc_joint",
    output_interpolator: str = "cubic",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build an unbiased FOD template with :func:`antstorch.syn.build_template`, reorienting the FODs.

    The registrations are those of :func:`fod_registration`, the template
    shape update reorients the SH template by the Jacobian of the update, and
    the Laplacian sharpening is applied per coefficient.

    Parameters
    ----------
    initial_template : ants.ANTsImage, optional
        Starting SH template (default: the weighted average of ``image_list``).
    image_list : list of ants.ANTsImage
        SH images with the same number of coefficients.
    lmax, directions, modulate, syn_metric, output_interpolator
        As in :func:`fod_registration`; ``output_interpolator`` also resamples
        the template in the shape update.
    **kwargs
        Forwarded to :func:`antstorch.syn.build_template` (``iterations``,
        ``extra_contrasts``, ``initial_alignment``, ``levels``, ``device``, ...).

    Returns
    -------
    dict
        :func:`antstorch.syn.build_template`'s result.
    """
    if image_list:
        images = list(image_list) + ([initial_template] if initial_template is not None else [])
        image_lmax = _check_fod_images(*images)
    else:
        image_lmax = 0  # build_template rejects the empty list
    _check_level_count(lmax, kwargs)
    reorientation = FODReorientation(image_lmax, lmax=lmax, directions=directions, modulate=modulate)
    return build_template(
        initial_template, image_list, reorientation=reorientation, syn_metric=syn_metric,
        output_interpolator=output_interpolator, **kwargs,
    )
