"""Fiber orientation distribution (FOD) registration support for :mod:`antstorch.syn`.

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
  ``diag(-1, -1, 1)`` before they act on fiber directions. The conversion is
  tested against ``mrtransform -warp ... -reorient_fod yes``.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .core.jacobian import _spatial_jacobian_nd
from .sh import FODReorienter, lmax_from_n, n_coefficients

__all__ = [
    "sh_lmax_of_image",
    "fod_image_read",
    "fod_image_write",
    "default_fod_lmax",
    "pullback_jacobian",
    "direction_transform",
    "FODRegistration",
]

_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def sh_lmax_of_image(image) -> int:
    """lmax of a multi-component ``ants.ANTsImage`` holding an even-degree SH series."""
    components = int(image.components)
    if components < 2:
        raise ValueError("an FOD image must have more than one component (SH coefficient)")
    return lmax_from_n(components)


def fod_image_read(filename: str):
    """Read an SH image written by MRtrix3 (a 4-D NIfTI) as a 3-D multi-component ``ants.ANTsImage``.

    ``ants.image_read`` returns 4-D NIfTI files as 4-D scalar images; this
    folds the fourth axis into the components so the image can be passed to
    :func:`antstorch.syn.syn_registration` with ``fod=True``. The SH
    coefficients are kept exactly as stored (MRtrix3 expresses them in the
    RAS scanner frame); only the geometry is converted to ITK's LPS metadata.
    """
    import ants

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
    """Write a multi-component ``ants.ANTsImage`` as a 4-D NIfTI (RAS affine) that MRtrix3 reads as an SH image."""
    import nibabel as nib

    if image.dimension != 3:
        raise ValueError("fod_image_write expects a 3-D multi-component image")
    array = np.asarray(image.numpy(), dtype=np.float32)
    if array.ndim == 3:
        array = array[..., None]
    direction = np.asarray(image.direction, dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(image.spacing, dtype=np.float64)
    origin = np.asarray(image.origin, dtype=np.float64)
    affine_lps = np.eye(4)
    affine_lps[:3, :3] = direction * spacing[None, :]
    affine_lps[:3, 3] = origin
    affine_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps
    nib.save(nib.Nifti1Image(np.ascontiguousarray(array), affine_ras), filename)


def default_fod_lmax(num_levels: int, image_lmax: int, max_lmax: int = 4) -> Tuple[int, ...]:
    """MRtrix3-like degree schedule: lmax 0 at the coarsest level, then 2, 4, ..., capped at ``min(max_lmax, image_lmax)``.

    ``mrregister`` registers FODs with lmax 0, 2 and 4 at scale factors 0.25,
    0.5 and 1.0; this reproduces that for three levels and keeps the cap for
    longer pyramids.
    """
    cap = min(int(max_lmax), int(image_lmax))
    cap -= cap % 2
    return tuple(min(2 * i, cap) for i in range(num_levels))


def _inverse_3x3(matrix: Tensor, eps: float = 1e-6) -> Tensor:
    """Batched 3x3 inverse by adjugate, with the determinant kept away from zero (folded voxels stay finite)."""
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
        [torch.stack([c00, c10, c20], dim=-1), torch.stack([c01, c11, c21], dim=-1), torch.stack([c02, c12, c22], dim=-1)],
        dim=-2,
    )
    return adjugate / det[..., None, None]


def pullback_jacobian(displacement: Tensor, spacing: Sequence[float], direction, affine: Optional[Tensor] = None) -> Tensor:
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


class FODRegistration:
    """FOD-specific state of one :func:`antstorch.syn.syn_registration` call.

    Parameters
    ----------
    image_lmax : int
        lmax of the input SH images.
    levels_lmax : sequence of int
        lmax used at each pyramid level (see :func:`default_fod_lmax`).
    directions : int, str or array
        aPSF directions for the reorientation (see :func:`antstorch.syn.sh.load_directions`).
    modulate_output : bool
        Apply FOD modulation (MRtrix3's ``-modulate fod``) to the warped output images.
    chunk_size : int
        Voxels per reorientation batch.
    sh_frame : str
        Frame of the SH basis, ``'RAS'`` for MRtrix3 images.
    extra_channels : int
        Number of non-SH channels (extra contrasts) appended after the SH
        coefficients of the images: they are registered at every level
        alongside the series and are never reoriented.
    device, dtype
        Where the reorientation matrices live.
    """

    def __init__(
        self,
        image_lmax: int,
        levels_lmax: Sequence[int],
        *,
        directions=60,
        modulate_output: bool = False,
        chunk_size: int = 1 << 14,
        sh_frame: str = "RAS",
        extra_channels: int = 0,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        self.image_lmax = int(image_lmax)
        self.levels_lmax = tuple(int(v) for v in levels_lmax)
        for lmax in self.levels_lmax:
            if lmax < 0 or lmax % 2 or lmax > self.image_lmax:
                raise ValueError(
                    f"per-level lmax values must be even and in [0, {self.image_lmax}] (the images' lmax), got {self.levels_lmax}"
                )
        self.directions = directions
        self.modulate_output = bool(modulate_output)
        self.chunk_size = int(chunk_size)
        self.sh_frame = sh_frame
        self.extra_channels = int(extra_channels)
        if self.extra_channels < 0:
            raise ValueError(f"extra_channels must be non-negative, got {extra_channels}")
        self.device = device
        self.dtype = dtype
        self._reorienters: Dict[int, FODReorienter] = {}

    def reorienter(self, lmax: int) -> Optional[FODReorienter]:
        """The (cached) :class:`FODReorienter` for ``lmax``; ``None`` for lmax 0 (nothing to reorient)."""
        if lmax < 2:
            return None
        if lmax not in self._reorienters:
            self._reorienters[lmax] = FODReorienter(lmax, self.directions, device=self.device, dtype=self.dtype)
        return self._reorienters[lmax]

    def channels(self, image: Tensor, lmax: int) -> Tensor:
        """The channels registered at a level: the SH series up to ``lmax`` plus the trailing extra-contrast channels."""
        sh_part = image[:, : n_coefficients(lmax)]
        if self.extra_channels == 0:
            return sh_part
        return torch.cat([sh_part, image[:, image.shape[1] - self.extra_channels :]], dim=1)

    @staticmethod
    def split(image: Tensor, lmax: int) -> Tuple[Tensor, Tensor]:
        """Split a ``[SH series up to lmax | extra channels]`` tensor into its SH and extra parts (the latter possibly empty)."""
        n = n_coefficients(lmax)
        return image[:, :n], image[:, n:]

    def _reorient(self, reorienter: FODReorienter, image: Tensor, transform: Tensor, lmax: int, modulate: bool = False) -> Tensor:
        """Reorient the SH part of ``image`` (series up to ``lmax``) and carry its extra channels through unchanged."""
        sh_part, extras = self.split(image, lmax)
        out = reorienter.reorient(sh_part, transform, modulate=modulate, chunk_size=self.chunk_size)
        return out if extras.shape[1] == 0 else torch.cat([out, extras], dim=1)

    @staticmethod
    def scalar_channel(image):
        """The ``l = 0`` coefficient of a multi-component ``ants.ANTsImage`` as a scalar image (for the affine stage)."""
        import ants

        return ants.split_channels(image)[0]

    @staticmethod
    def intensity_scale(image: Tensor, percentile: float = 99.0) -> float:
        """``1 / percentile`` of the positive ``l = 0`` coefficients: one factor applied to every channel of both images."""
        l0 = image[:, 0]
        positive = l0[l0 > 0].detach()
        if positive.numel() == 0:
            return 1.0
        if positive.numel() > 4_000_000:
            index = torch.randint(0, positive.numel(), (4_000_000,), device=positive.device)
            positive = positive[index]
        value = float(torch.quantile(positive.float(), percentile / 100.0))
        return 1.0 / value if value > 0 else 1.0

    def midpoint_hook(self, lmax: int, fixed_meta: Dict[str, tuple], M_phys: Tensor) -> Optional[Callable]:
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
        reorienter = self.reorienter(lmax)
        if reorienter is None:
            return None
        spacing, direction = fixed_meta["spacing"], fixed_meta["direction"]
        sh_frame = self.sh_frame

        def hook(I_mid: Tensor, J_mid: Tensor, warp_l2r: Tensor, warp_r2l: Tensor) -> Tuple[Tensor, Tensor]:
            with torch.no_grad():
                transform_fixed = direction_transform(pullback_jacobian(warp_l2r.detach(), spacing, direction), sh_frame)
                transform_moving = direction_transform(
                    pullback_jacobian(warp_r2l.detach(), spacing, direction, affine=M_phys.detach()), sh_frame
                )
            return (
                self._reorient(reorienter, I_mid, transform_fixed, lmax),
                self._reorient(reorienter, J_mid, transform_moving, lmax),
            )

        return hook

    def reorient_warped(self, image: Tensor, displacement: Tensor, meta: Dict[str, tuple], modulate: Optional[bool] = None) -> Tensor:
        """Reorient a full-lmax image sampled through ``x + displacement`` on the grid described by ``meta``."""
        reorienter = self.reorienter(self.image_lmax)
        if reorienter is None:
            return image
        with torch.no_grad():
            transform = direction_transform(pullback_jacobian(displacement, meta["spacing"], meta["direction"]), self.sh_frame)
        return self._reorient(reorienter, image, transform, self.image_lmax, self.modulate_output if modulate is None else modulate)

    def reorient_affine(self, image: Tensor, matrix_xyz: Tensor, modulate: Optional[bool] = None) -> Tensor:
        """Reorient a full-lmax image resampled through the pull-back affine ``x -> matrix_xyz x + t`` (ITK ``(x, y, z)`` order)."""
        reorienter = self.reorienter(self.image_lmax)
        if reorienter is None:
            return image
        jacobian = matrix_xyz.detach().to(device=image.device, dtype=image.dtype)
        transform = direction_transform(jacobian, self.sh_frame)
        return self._reorient(reorienter, image, transform, self.image_lmax, self.modulate_output if modulate is None else modulate)
