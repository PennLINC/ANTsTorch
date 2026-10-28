"""Real spherical harmonics, apodized point spread functions and FOD reorientation.

Everything here follows the MRtrix3 conventions (``core/math/SH.h``,
``core/math/legendre.h`` and ``core/registration/transform/reorient.h`` of
MRtrix3), so that spherical harmonic (SH) images written by ``dwi2fod``,
``amp2sh`` or ``mtnormalise`` can be consumed directly and the reorientation
reproduces ``mrtransform -reorient_fod yes`` and ``mrregister``:

* only even harmonic degrees ``l = 0, 2, ..., lmax`` are stored, with the
  coefficient of order ``m`` of degree ``l`` at index ``l (l + 1) / 2 + m``,
  so a series up to ``lmax`` has ``(lmax + 1) (lmax + 2) / 2`` coefficients;
* the basis functions are the orthonormal real harmonics
  ``Y_l0 = N_l0 P_l(cos theta)``,
  ``Y_lm = sqrt(2) N_lm P_l^m(cos theta) cos(m phi)`` for ``m > 0`` and
  ``Y_lm = sqrt(2) N_l|m| P_l^|m|(cos theta) sin(|m| phi)`` for ``m < 0``,
  where the associated Legendre functions carry the Condon-Shortley phase;
* ``cos theta = z`` and ``phi = atan2(y, x)``;
* the SH basis is expressed in the image's scanner (world) frame, which for
  MRtrix3 and NIfTI is RAS, not in voxel axes and not in ITK's LPS frame.

FOD reorientation uses the apodized point spread function (aPSF) method of
Raffelt et al., NeuroImage 2012: an FOD is decomposed into a fixed set of
aPSFs by a pseudo-inverse, each aPSF direction is transformed by the local
linear map and normalized, and the aPSFs are recomposed at the new directions.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor

__all__ = [
    "n_coefficients",
    "lmax_from_n",
    "sh_index",
    "coefficient_degrees",
    "sh_basis",
    "apsf_rh",
    "apsf_matrix",
    "load_directions",
    "spherical_to_cartesian",
    "FODReorienter",
]

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Rotational harmonic coefficients of the apodized point spread function per
# lmax, from MRtrix3's Math::SH::aPSF (core/math/SH.h). Entry i scales degree
# l = 2 i.
_APSF_RH = {
    2: (1.00000000, 0.41939279),
    4: (1.00000000, 0.63608543, 0.18487087),
    6: (1.00000000, 0.75490341, 0.37126442, 0.09614699),
    8: (1.00000000, 0.82384816, 0.51261696, 0.22440563, 0.05593079),
    10: (1.00000000, 0.86725945, 0.61519436, 0.34570667, 0.14300355, 0.03548062),
    12: (1.00000000, 0.89737759, 0.69278503, 0.45249879, 0.24169922, 0.09826171, 0.02502481),
    14: (1.00000000, 0.91717853, 0.74685644, 0.53467773, 0.33031863, 0.17013825, 0.06810155, 0.01754930),
    16: (1.00000000, 0.93261196, 0.79064858, 0.60562880, 0.41454703, 0.24880754, 0.12661242, 0.05106681, 0.01365433),
}

_INV_SQRT_4PI = 0.282094791773878  # 1 / sqrt(4 pi), as written in MRtrix3


def n_coefficients(lmax: int) -> int:
    """Number of SH coefficients of an even-degree series up to ``lmax``."""
    if lmax < 0 or lmax % 2:
        raise ValueError(f"lmax must be a non-negative even integer, got {lmax}")
    return (lmax + 1) * (lmax + 2) // 2


def lmax_from_n(n: int) -> int:
    """Largest even ``lmax`` whose series has exactly ``n`` coefficients (``ValueError`` otherwise)."""
    if n < 1:
        raise ValueError(f"the number of SH coefficients must be positive, got {n}")
    lmax = 2 * int(math.floor((math.sqrt(1.0 + 8.0 * n) - 3.0) / 4.0))
    if n_coefficients(lmax) != n:
        raise ValueError(f"{n} volumes is not the size of an even-degree SH series (1, 6, 15, 28, 45, ...)")
    return lmax


def sh_index(l: int, m: int) -> int:
    """Index of the ``(l, m)`` coefficient in the MRtrix3 ordering."""
    return l * (l + 1) // 2 + m


def coefficient_degrees(lmax: int, device=None) -> Tensor:
    """Degree ``l`` of every coefficient index, as an integer tensor of length ``n_coefficients(lmax)``."""
    degrees = torch.empty(n_coefficients(lmax), dtype=torch.long, device=device)
    for l in range(0, lmax + 1, 2):
        degrees[sh_index(l, -l) : sh_index(l, l) + 1] = l
    return degrees


def _legendre_sph_column(lmax: int, m: int, z: Tensor, sin_theta: Tensor) -> dict:
    """Normalized associated Legendre functions ``P_l^m(z)`` for ``l = m .. lmax``.

    Transcription of MRtrix3's ``Legendre::Plm_sph`` (core/math/legendre.h),
    including its Condon-Shortley phase, vectorized over ``z``.
    """
    v0 = torch.full_like(z, _INV_SQRT_4PI)
    if m:
        # sqrt((2 m + 1) * prod_{k=1}^{m} (2 k - 1) / (2 k)) * sin(theta)^m
        ratio = 1.0
        for k in range(1, m + 1):
            ratio *= (2.0 * k - 1.0) / (2.0 * k)
        v0 = v0 * (math.sqrt((2.0 * m + 1.0) * ratio) * sin_theta.pow(m))
        if m & 1:
            v0 = -v0
    values = {m: v0}
    if m == lmax:
        return values
    f = math.sqrt(2.0 * m + 3.0)
    v1 = z * (f * v0)
    values[m + 1] = v1
    for n in range(m + 2, lmax + 1):
        v2 = z * v1 - v0 / f
        f = math.sqrt((4.0 * n * n - 1.0) / (n * n - m * m))
        v0 = v1
        v1 = f * v2
        values[n] = v1
    return values


def sh_basis(directions: Tensor, lmax: int) -> Tensor:
    """Evaluate the real SH basis at unit ``directions``.

    Parameters
    ----------
    directions : Tensor
        ``(..., 3)`` unit vectors ``(x, y, z)`` in the frame the SH series is
        expressed in.
    lmax : int
        Maximum (even) harmonic degree.

    Returns
    -------
    Tensor
        ``(..., n_coefficients(lmax))``; ``amplitudes = sh_basis(d, lmax) @ coefficients``
        reproduces MRtrix3's ``sh2amp``.
    """
    n = n_coefficients(lmax)
    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    rxy = torch.hypot(x, y)
    nonzero = rxy > 0
    cos_phi = torch.where(nonzero, x / rxy.clamp_min(1e-30), torch.ones_like(x))
    sin_phi = torch.where(nonzero, y / rxy.clamp_min(1e-30), torch.zeros_like(y))
    sin_theta = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    out = torch.empty(directions.shape[:-1] + (n,), dtype=directions.dtype, device=directions.device)
    sqrt2 = math.sqrt(2.0)
    cos_m, sin_m = torch.ones_like(x), torch.zeros_like(x)  # cos(m phi), sin(m phi) by recurrence
    for m in range(0, lmax + 1):
        legendre = _legendre_sph_column(lmax, m, z, sin_theta)
        if m == 0:
            for l in range(0, lmax + 1, 2):
                out[..., sh_index(l, 0)] = legendre[l]
            continue
        cos_m, sin_m = cos_m * cos_phi - sin_m * sin_phi, sin_m * cos_phi + cos_m * sin_phi
        for l in range(m + (m & 1), lmax + 1, 2):
            out[..., sh_index(l, m)] = sqrt2 * legendre[l] * cos_m
            out[..., sh_index(l, -m)] = sqrt2 * legendre[l] * sin_m
    return out


def apsf_rh(lmax: int, device=None, dtype=torch.float32) -> Tensor:
    """Rotational harmonic coefficients of the apodized PSF for ``lmax`` (one per even degree)."""
    if lmax not in _APSF_RH:
        raise ValueError(f"no apodized PSF coefficients for lmax {lmax} (available: {sorted(_APSF_RH)})")
    return torch.tensor(_APSF_RH[lmax], device=device, dtype=dtype)


def apsf_matrix(directions: Tensor, lmax: int) -> Tensor:
    """The ``(n_coefficients, n_directions)`` matrix of SH coefficients of one apodized PSF per direction.

    Equals MRtrix3's ``aPSF_weights_to_FOD_transform``: the SH delta function at
    each direction, spherically convolved with the aPSF kernel.
    """
    basis = sh_basis(directions, lmax)  # (n_directions, n_coefficients)
    degrees = coefficient_degrees(lmax, directions.device)
    gain = apsf_rh(lmax, device=directions.device, dtype=directions.dtype)[degrees // 2]
    return (basis * gain).transpose(0, 1).contiguous()


def spherical_to_cartesian(az_el: Union[Tensor, np.ndarray]) -> Tensor:
    """Convert ``(..., 2)`` ``[azimuth, inclination]`` rows to unit ``(..., 3)`` vectors (MRtrix3 convention)."""
    az_el = torch.as_tensor(np.asarray(az_el), dtype=torch.float64)
    az, el = az_el[..., 0], az_el[..., 1]
    return torch.stack([torch.sin(el) * torch.cos(az), torch.sin(el) * torch.sin(az), torch.cos(el)], dim=-1)


def load_directions(
    source: Union[str, int, Tensor, np.ndarray, Sequence] = 60, *, device=None, dtype=torch.float32
) -> Tensor:
    """Load a direction set as ``(n, 3)`` unit vectors.

    ``source`` may be the integer ``60`` or ``300`` (the electrostatic-repulsion
    sets shipped in ``antstorch/syn/data``, generated with MRtrix3's ``dirgen``),
    a path to a text file with ``[azimuth inclination]`` or ``[x y z]`` rows
    (``#`` comments allowed, MRtrix3 ``dirgen``/``-directions`` format), or an
    array of such rows.
    """
    if isinstance(source, (int, np.integer)):
        path = os.path.join(_DATA_DIR, f"directions_{int(source)}.txt")
        if not os.path.exists(path):
            raise ValueError(f"no bundled direction set with {source} directions (bundled: 60, 300)")
        source = path
    if isinstance(source, str):
        rows = np.loadtxt(source, comments="#", ndmin=2)
    else:
        if isinstance(source, Tensor):
            source = source.detach().cpu().numpy()
        rows = np.asarray(source, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] not in (2, 3):
        raise ValueError("directions must be an (n, 2) [azimuth inclination] or (n, 3) [x y z] array")
    if rows.shape[1] == 2:
        directions = spherical_to_cartesian(rows)
    else:
        directions = torch.as_tensor(rows, dtype=torch.float64)
        directions = directions / directions.norm(dim=-1, keepdim=True)
    return directions.to(device=device, dtype=dtype)


class FODReorienter:
    """Apodized-PSF reorientation of SH FOD images under a local linear map.

    The ``transforms`` passed to :meth:`reorient` map fiber directions of the
    input FOD to directions of the output FOD, in the frame the SH series is
    expressed in. For a pull-back (output to input) deformation ``phi`` this
    is the inverse of its Jacobian, ``(d phi / d x)^-1``, as in MRtrix3's
    ``Registration::Transform::reorient``. Registrations use it through
    :class:`antstorch.syn.fod.FODReorientation`.

    Parameters
    ----------
    lmax : int
        Harmonic degree of the SH series to reorient (the series must have
        exactly ``n_coefficients(lmax)`` channels).
    directions : int, str, array or Tensor
        aPSF directions, see :func:`load_directions`. MRtrix3 uses 60
        directions in ``mrregister`` and 300 in ``mrtransform``.
    device, dtype
        Where the precomputed matrices live.
    """

    def __init__(self, lmax: int, directions=60, *, device=None, dtype: torch.dtype = torch.float32):
        if lmax < 2:
            raise ValueError("FOD reorientation needs lmax >= 2 (an lmax 0 series is isotropic)")
        self.lmax = int(lmax)
        self.n_coefficients = n_coefficients(self.lmax)
        self.directions = load_directions(directions, device=device, dtype=torch.float64)
        if self.directions.shape[0] < self.n_coefficients:
            raise ValueError(
                f"{self.directions.shape[0]} aPSF directions cannot represent an lmax {self.lmax} series "
                f"({self.n_coefficients} coefficients); use more directions"
            )
        forward = apsf_matrix(self.directions, self.lmax)  # (n_coefficients, n_directions), float64
        self.fod_to_weights = torch.linalg.pinv(forward).to(dtype=dtype)  # (n_directions, n_coefficients)
        self.directions = self.directions.to(dtype=dtype)
        degrees = coefficient_degrees(self.lmax, device)
        self.band_gain = apsf_rh(self.lmax, device=device, dtype=dtype)[degrees // 2]
        self.device = self.fod_to_weights.device
        self.dtype = dtype

    def to(self, device=None, dtype=None) -> "FODReorienter":
        """Move the precomputed matrices to ``device``/``dtype``; returns ``self``."""
        self.fod_to_weights = self.fod_to_weights.to(device=device, dtype=dtype)
        self.directions = self.directions.to(device=device, dtype=dtype)
        self.band_gain = self.band_gain.to(device=device, dtype=dtype)
        self.device = self.fod_to_weights.device
        self.dtype = self.fod_to_weights.dtype
        return self

    def reorient_coefficients(self, coefficients: Tensor, transforms: Tensor, *, modulate: bool = False) -> Tensor:
        """Reorient a batch of SH series.

        Parameters
        ----------
        coefficients : Tensor
            ``(n, n_coefficients)`` SH coefficients.
        transforms : Tensor
            ``(n, 3, 3)`` or ``(3, 3)`` direction maps (see the class notes).
        modulate : bool
            Scale each aPSF by ``|T d| / det(T)`` (MRtrix3's ``-modulate fod``),
            which preserves the apparent fiber density through volume changes.

        Returns
        -------
        Tensor
            ``(n, n_coefficients)`` reoriented coefficients.
        """
        if coefficients.shape[-1] != self.n_coefficients:
            raise ValueError(f"expected {self.n_coefficients} coefficients per voxel, got {coefficients.shape[-1]}")
        transforms = transforms.to(dtype=coefficients.dtype)
        if transforms.ndim == 2:
            transforms = transforms.unsqueeze(0).expand(coefficients.shape[0], 3, 3)
        weights = coefficients @ self.fod_to_weights.transpose(0, 1).to(coefficients.dtype)  # (n, n_directions)
        new_directions = torch.einsum("nij,dj->ndi", transforms, self.directions.to(coefficients.dtype))
        norms = new_directions.norm(dim=-1)  # (n, n_directions)
        new_directions = new_directions / norms.clamp_min(1e-12).unsqueeze(-1)
        if modulate:
            weights = weights * (norms / torch.linalg.det(transforms).unsqueeze(-1))
        basis = sh_basis(new_directions, self.lmax)  # (n, n_directions, n_coefficients)
        return torch.einsum("nd,ndk->nk", weights, basis) * self.band_gain.to(coefficients.dtype)

    def reorient(
        self,
        image: Tensor,
        transforms: Tensor,
        *,
        mask: Optional[Tensor] = None,
        modulate: bool = False,
        chunk_size: int = 1 << 14,
        checkpoint: Optional[bool] = None,
    ) -> Tensor:
        """Reorient a channel-first SH image.

        Parameters
        ----------
        image : Tensor
            ``(B, n_coefficients, *spatial)`` SH image.
        transforms : Tensor
            ``(B, *spatial, 3, 3)`` per-voxel direction maps, or a single ``(3, 3)``.
        mask : Tensor, optional
            ``(B, 1, *spatial)``; only voxels with ``mask > 0.5`` are reoriented.
            Voxels whose ``l = 0`` coefficient is not positive are never
            reoriented (they contain no FOD), matching MRtrix3.
        modulate : bool
            See :meth:`reorient_coefficients`.
        chunk_size : int
            Voxels per batch; bounds the ``chunk_size x n_directions x n_coefficients``
            working memory.
        checkpoint : bool, optional
            Recompute the per-chunk aPSF bases in the backward pass instead of
            storing them (``torch.utils.checkpoint``); defaults to ``True`` when
            ``image`` requires a gradient.

        Returns
        -------
        Tensor
            Reoriented image, same shape as ``image``.
        """
        if image.ndim < 3 or image.shape[1] != self.n_coefficients:
            raise ValueError(f"image must be (B, {self.n_coefficients}, *spatial), got {tuple(image.shape)}")
        batch, channels = image.shape[:2]
        spatial = image.shape[2:]
        flat = image.movedim(1, -1).reshape(-1, channels)  # (B * voxels, C)
        select = flat[:, 0] > 0
        if mask is not None:
            select = select & (mask.reshape(-1) > 0.5)
        indices = torch.nonzero(select, as_tuple=False).squeeze(-1)
        out = flat.clone()
        if indices.numel() == 0:
            return out.reshape(batch, *spatial, channels).movedim(-1, 1)
        if transforms.ndim == 2:
            flat_transforms = None
        else:
            if tuple(transforms.shape[:-2]) != (batch,) + tuple(spatial):
                raise ValueError(
                    f"transforms must be (B, *spatial, 3, 3) matching the image, got {tuple(transforms.shape)}"
                )
            flat_transforms = transforms.reshape(-1, 3, 3)
        if checkpoint is None:
            checkpoint = bool(image.requires_grad)
        results = []
        for start in range(0, indices.numel(), chunk_size):
            chunk = indices[start : start + chunk_size]
            coefficients = flat[chunk]
            chunk_transforms = transforms if flat_transforms is None else flat_transforms[chunk]
            if checkpoint and coefficients.requires_grad:
                reoriented = torch.utils.checkpoint.checkpoint(
                    lambda c, t: self.reorient_coefficients(c, t, modulate=modulate),
                    coefficients,
                    chunk_transforms,
                    use_reentrant=False,
                )
            else:
                reoriented = self.reorient_coefficients(coefficients, chunk_transforms, modulate=modulate)
            results.append(reoriented)
        out = out.index_put((indices,), torch.cat(results, dim=0))
        return out.reshape(batch, *spatial, channels).movedim(-1, 1)
