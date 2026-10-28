"""Unbiased group-template construction with ANTsTorch SyN.

Port of ``syntx.template.build_template`` (itself a port of
``ants.build_template``) onto :func:`antstorch.syn.syn_registration`.  Scope of
this port (decided 2026-10-08): the SyN / SyNOnly path of the PyTorch backend
only.  ``syntx``'s one-directional ``type_of_transform='Greedy'`` mode
(``syntx.greedy`` is not ported) and its ``backend='ants'`` fallback
(``ants.build_template`` already covers that) are intentionally left out.

Each iteration registers every image to the current template, averages the
warped images, then moves that average back by ``gradient_step`` times the mean
(negated) warp and the inverse of the average affine, so the template drifts
toward the centre of shape of the group.  Per iteration the size and roughness
of the mean warp (RMS displacement, membrane and bending energy) and the mean
absolute template change are recorded.  An optional linear pre-alignment
(``initial_alignment``, the ANTs script's ``-r 1``) replaces the plain average
of the unaligned inputs as the starting template.

Multi-component templates are updated in torch and sharpened per channel;
with a ``reorientation`` (see :func:`antstorch.syn.syn_registration`) the shape
update also reorients the template by the Jacobian of the update, which
``ants.apply_transforms`` cannot do. :func:`antstorch.syn.build_fod_template`
builds FOD templates this way.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import ants
import numpy as np
import torch

from ..ants_transform_io import read_affine_transform
from .bridge import (
    _same_grid,
    ants_image_metadata,
    ants_image_to_tensor,
    flip_affine_xyz_to_zyx,
    metadata_tensors_from_dict,
    tensor_to_ants_image,
)
from .core.grid import get_physical_grid_torch, grid_sample_nd, physical_to_normalized_torch_cached
from .syn import _LINEAR_TRANSFORM_TYPES, _OUTPUT_INTERPOLATORS, _resolve_device, syn_registration

__all__ = ["build_template"]

_MANAGED_KWARGS = ("initial_affine", "outprefix")
# grid_sample_nd modes of syn_registration's output interpolators.
_INTERPOLATOR_MODES = {"linear": "bilinear", "nearest": "nearest", "cubic": "cubic", "bspline": "bspline"}


def _sample_like_itk(
    tensor: torch.Tensor, coords: torch.Tensor, shape_t: torch.Tensor, interpolator: str = "linear"
) -> torch.Tensor:
    """Sampling with ITK's (and MRtrix3's) boundary semantics (what ``ants.apply_transforms`` does).

    ITK's interpolators clamp neighbor indices to the buffer, so points within
    half a voxel beyond the outermost voxel centers get the edge value
    ("border" padding), and points further out get the default value 0.
    ``coords`` are normalized grid_sample coordinates (``(x, y, z)`` order);
    ``shape_t`` is the torch (reversed) spatial shape; ``interpolator`` is one
    of ``'linear'``, ``'cubic'`` (interpolating Catmull-Rom, MRtrix3's
    ``cubic``), ``'bspline'`` or ``'nearest'``.
    """
    mode = _INTERPOLATOR_MODES[interpolator]
    sampled = grid_sample_nd(
        tensor, coords, mode=mode, padding_mode="border", align_corners=True, interpolator=mode
    )
    dim = coords.shape[-1]
    inside = None
    for d in range(dim):
        size = shape_t[dim - 1 - d].to(coords.dtype)
        limit = 1.0 + 1.0 / (size - 1.0).clamp_min(1.0)
        axis_inside = coords[..., d].abs() <= limit
        inside = axis_inside if inside is None else (inside & axis_inside)
    return sampled * inside.unsqueeze(1).to(sampled.dtype)


def _resample_vector_onto(
    image: ants.ANTsImage, target: ants.ANTsImage, device="cpu", interpolator: str = "linear"
) -> ants.ANTsImage:
    """Resampling of a (multi-component) image onto ``target``'s grid through the identity map.

    ``ants.resample_image_to_target`` crashes on multi-component images (ANTsPy
    0.6.3), so the resampling is done in torch; no reorientation is needed for
    the identity map, so this is valid for SH images too.
    """
    if _same_grid(image, target):
        return image.clone()
    source_meta = ants_image_metadata(image)
    target_meta = ants_image_metadata(target)
    tensor = ants_image_to_tensor(image, device, torch.float32, normalize=False)
    X = get_physical_grid_torch(
        target_meta["torch_shape"], target_meta["spacing"], target_meta["origin"], target_meta["direction"],
        device=device,
    )
    mt = metadata_tensors_from_dict(source_meta, torch.device(device), torch.float32)
    coords = physical_to_normalized_torch_cached(X, mt["shape_t"], mt["spacing_t"], mt["origin_t"], mt["direction_t"])
    resampled = _sample_like_itk(tensor, coords, mt["shape_t"], interpolator)
    return tensor_to_ants_image(resampled, target)


def _apply_transforms_torch(
    image: ants.ANTsImage,
    scaled_warp: Optional[ants.ANTsImage],
    affine_fn: Optional[str],
    interpolator: str = "linear",
    device="cpu",
    reorientation=None,
) -> ants.ANTsImage:
    """``ants.apply_transforms`` of a template through its shape update, in torch.

    Equivalent to ``ants.apply_transforms(fixed=image, moving=image, [warp,
    affine], whichtoinvert=[0, 1])`` for multi-component images and
    non-linear interpolators: the image is sampled at
    ``phi(x) = A^-1 ((x + w(x)) - t)`` (the warp first, then the inverted
    average affine, ANTs' order for the transform list) with ``interpolator``
    (see :func:`_sample_like_itk`). With a ``reorientation`` the channels are
    also reoriented by the Jacobian of ``phi``, as the warped outputs of
    :func:`antstorch.syn.syn_registration` are.
    """
    meta = ants_image_metadata(image)
    tensor = ants_image_to_tensor(image, device, torch.float32, normalize=False)  # (1, C, z, y, x)
    X = get_physical_grid_torch(meta["torch_shape"], meta["spacing"], meta["origin"], meta["direction"], device=device)
    if scaled_warp is not None:
        warp_xyz = ants_image_to_tensor(scaled_warp, device, torch.float32, normalize=False)[0]  # (3, z, y, x)
        displacement = warp_xyz.movedim(0, -1).flip(-1).unsqueeze(0)  # (1, z, y, x, 3), (z, y, x) components
    else:
        displacement = torch.zeros_like(X)
    points = X + displacement
    affine_zyx = None
    if affine_fn is not None:
        matrix, translation = read_affine_transform(affine_fn, 3)  # y = M x + t in ITK physical coordinates
        inverse_matrix = np.linalg.inv(matrix)
        inverse_translation = -inverse_matrix @ translation
        affine_zyx, translation_zyx = flip_affine_xyz_to_zyx(
            torch.as_tensor(inverse_matrix, dtype=torch.float32),
            torch.as_tensor(inverse_translation, dtype=torch.float32),
        )
        affine_zyx = affine_zyx.to(device)
        points = points @ affine_zyx.transpose(0, 1) + translation_zyx.to(device)
    mt = metadata_tensors_from_dict(meta, torch.device(device), torch.float32)
    coords = physical_to_normalized_torch_cached(
        points, mt["shape_t"], mt["spacing_t"], mt["origin_t"], mt["direction_t"]
    )
    warped = _sample_like_itk(tensor, coords, mt["shape_t"], interpolator)
    if reorientation is not None:
        warped = reorientation.reorient_warped(warped, displacement, meta, affine=affine_zyx)
    return tensor_to_ants_image(warped, image)


def _scalar_reference(image: ants.ANTsImage) -> ants.ANTsImage:
    """``image`` itself, or its first channel for a multi-component image (a scalar image on the same grid)."""
    return ants.split_channels(image)[0] if image.components > 1 else image


def _prepare_template_extras(
    image_list: List[ants.ANTsImage], extra_contrasts, interpolator: str, device
) -> List[List[ants.ANTsImage]]:
    """The extra contrasts of :func:`build_template` as lists of scalar images on their primary images' grids.

    ``extra_contrasts[c][k]`` belongs to ``image_list[k]``; an image on another grid (a 1 mm
    T1-weighted volume next to a 1.8 mm FOD image, say) is resampled onto the primary grid with
    ``interpolator``, since the template shape update shares one grid between the contrasts.
    """
    extras: List[List[ants.ANTsImage]] = []
    for c, contrast in enumerate(extra_contrasts or ()):
        contrast = list(contrast)
        if len(contrast) != len(image_list):
            raise ValueError(
                f"extra_contrasts[{c}] must hold one image per entry of image_list ({len(image_list)}), "
                f"got {len(contrast)}"
            )
        resampled = []
        for image, primary in zip(contrast, image_list):
            if image.components != 1:
                raise ValueError(
                    f"extra_contrasts[{c}]: extra contrasts must be scalar images, got {image.components} components"
                )
            target = _scalar_reference(primary)
            resampled.append(_resample_vector_onto(image, target, device=device, interpolator=interpolator))
        extras.append(resampled)
    return extras


def _apply_shape_update(
    image: ants.ANTsImage,
    scaled_warp: Optional[ants.ANTsImage],
    warp_fn: Optional[str],
    affine_fn: Optional[str],
    use_torch: bool,
    interpolator: str,
    device,
    reorientation=None,
) -> ants.ANTsImage:
    """Move a template (or an extra-contrast template) by the scaled mean warp and the inverted average affine."""
    if scaled_warp is None and affine_fn is None:
        return image
    if use_torch:
        return _apply_transforms_torch(image, scaled_warp, affine_fn, interpolator, device, reorientation)
    transforms = ([warp_fn] if warp_fn is not None else []) + ([affine_fn] if affine_fn is not None else [])
    invert = ([0] if warp_fn is not None else []) + ([1] if affine_fn is not None else [])
    return ants.apply_transforms(fixed=image, moving=image, transformlist=transforms, whichtoinvert=invert)


def _sharpen_image(image: ants.ANTsImage, blending_weight: float) -> ants.ANTsImage:
    """``w * image + (1 - w) * Sharpen(image)``, per component for a multi-component image.

    ``ants.iMath`` Sharpen is scalar only.
    """
    if image.components == 1:
        return image * blending_weight + ants.iMath(image, "Sharpen") * (1.0 - blending_weight)
    array = np.asarray(image.numpy(), dtype=np.float32)
    out = np.empty_like(array)
    for c in range(array.shape[-1]):
        if not np.isfinite(array[..., c]).all() or array[..., c].max() == array[..., c].min():
            # ITK's Laplacian sharpening rescales by the intensity range: NaN on a constant channel.
            out[..., c] = array[..., c]
            continue
        channel = ants.from_numpy(
            np.ascontiguousarray(array[..., c]), origin=image.origin, spacing=image.spacing,
            direction=image.direction,
        )
        sharpened = ants.iMath(channel, "Sharpen").numpy()
        out[..., c] = blending_weight * array[..., c] + (1.0 - blending_weight) * sharpened
    return ants.from_numpy(
        np.ascontiguousarray(out), origin=image.origin, spacing=image.spacing, direction=image.direction,
        has_components=True,
    )


def _normalize_weights(weights: Optional[Sequence[float]], n: int) -> np.ndarray:
    """Normalise per-image weights so they sum to 1 (uniform when ``None``)."""
    if weights is None:
        return np.ones(n, dtype=np.float64) / n
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"len(weights)={w.size} must equal len(image_list)={n}")
    if np.any(w < 0):
        raise ValueError("All weights must be non-negative.")
    total = w.sum()
    if total <= 0:
        raise ValueError("Weights must have a positive sum.")
    return w / total


def _initial_alignment_reference(reference, image_list, extra_contrasts):
    """Resolve ``initial_alignment_reference`` to ``(primary, extras or None, name, is_input)``.

    The primary is ``None`` for the average of the unaligned inputs.
    """
    n_extra = len(extra_contrasts or ())
    if reference is None:
        return None, None, "the average of the unaligned inputs", False
    if isinstance(reference, (int, np.integer)) and not isinstance(reference, bool):
        if not 0 <= reference < len(image_list):
            raise ValueError(
                f"initial_alignment_reference {reference} is not an index into image_list ({len(image_list)} images)"
            )
        extras = [contrast[reference] for contrast in extra_contrasts] if n_extra else None
        return image_list[reference], extras, f"image {reference}", True
    if isinstance(reference, ants.ANTsImage):
        if n_extra:
            raise ValueError(
                "with extra contrasts, initial_alignment_reference must be a sequence: "
                "the primary reference followed by one image per extra contrast"
            )
        return reference, None, "the given reference image", False
    images = list(reference)
    if len(images) != 1 + n_extra or not all(isinstance(image, ants.ANTsImage) for image in images):
        raise ValueError(
            f"initial_alignment_reference must be an image index, an image, or a sequence of {1 + n_extra} images "
            f"(primary + extra contrasts), got {reference!r}"
        )
    return images[0], (images[1:] or None), "the given reference images", False


def _initial_template(
    image_list: List[ants.ANTsImage], weights: Sequence[float], interpolator: str = "linear"
) -> ants.ANTsImage:
    """Weighted average of ``image_list``, resampled onto the first image."""
    template = image_list[0] * 0
    for image, weight in zip(image_list, weights):
        if image.components > 1 or interpolator != "linear":
            template = template + _resample_vector_onto(image, template, interpolator=interpolator) * float(weight)
        else:
            template = template + ants.resample_image_to_target(image * float(weight), template)
    return template


def _template_change(old: ants.ANTsImage, new: ants.ANTsImage) -> float:
    """Mean absolute voxel-intensity change between two templates."""
    return float(np.mean(np.abs(old.numpy().astype(np.float64) - new.numpy().astype(np.float64))))


def _compute_shape_residual(mean_warp: ants.ANTsImage) -> Dict[str, float]:
    """Size and roughness of the mean warp field.

    ``l2_norm`` is the RMS displacement magnitude (physical units),
    ``membrane_energy`` the mean squared Frobenius norm of the Jacobian of the
    displacement and ``bending_energy`` that of its Hessian.
    """
    arr = mean_warp.numpy().astype(np.float64)  # (*spatial, dim)
    spacing = [float(s) for s in mean_warp.spacing]
    dim = arr.shape[-1]

    l2_norm = float(np.sqrt(np.mean(np.sum(arr**2, axis=-1))))

    membrane = 0.0
    bending = 0.0
    for k in range(dim):
        component = arr[..., k]
        for j in range(dim):
            grad_j = np.gradient(component, spacing[j], axis=j)
            membrane += float(np.mean(grad_j**2))
            for i in range(dim):
                bending += float(np.mean(np.gradient(grad_j, spacing[i], axis=i) ** 2))
    return {"l2_norm": l2_norm, "membrane_energy": membrane, "bending_energy": bending}


def _polar_rotation(matrix: np.ndarray) -> np.ndarray:
    """Rotation factor ``U V^T`` of the polar decomposition of a square matrix (SVD form)."""
    u, _, vt = np.linalg.svd(matrix)
    return u @ vt


def _apply_affine_to_vector_image(fixed, moving, affine_file: str, invert: bool = True):
    """NumPy equivalent of ``ants.apply_transforms(fixed, moving, imagetype=1, transformlist=affine_file, whichtoinvert=[invert])``.

    ``antsApplyTransforms`` resamples every component at the transformed location and rotates the vector by the rotation
    factor of the polar decomposition of the affine's linear part (``M`` for an inverted transform, ``M^-1`` otherwise),
    applied to the raw components with no spacing / direction conversion. Reproduced here exactly (agreement to ~1e-8
    against ANTs 0.6.3 on Linux, 2D and 3D, with anisotropic spacing, a rotated direction and a non-zero centre).

    Used as a fallback because some ANTsPy builds (the macOS 0.6.3 wheels) cannot read a multi-component image in
    ``antsApplyTransforms`` ("Unable to read vector image"), for any dimension and any ``imagetype``.
    """
    dim = fixed.dimension
    transform = ants.read_transform(affine_file)
    matrix = np.asarray(transform.parameters[: dim * dim], dtype=np.float64).reshape(dim, dim)
    rotation = _polar_rotation(matrix if invert else np.linalg.inv(matrix))
    channels = ants.split_channels(moving)
    resampled = [
        ants.apply_transforms(fixed=fixed, moving=channel, transformlist=[affine_file], whichtoinvert=[bool(invert)])
        for channel in channels
    ]
    stacked = np.stack([r.numpy() for r in resampled], axis=-1)
    rotated = np.einsum("ij,...j->...i", rotation, stacked).astype(np.float32)
    out = ants.from_numpy(
        rotated, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction, has_components=True
    )
    return out.clone(moving.pixeltype)


def _apply_affine_to_vectors(fixed, moving, affine_file: str, invert: bool = True):
    """``ants.apply_transforms(..., imagetype=1)`` of a vector image through an affine, with the NumPy fallback."""
    try:
        return ants.apply_transforms(
            fixed=fixed, moving=moving, imagetype=1, transformlist=affine_file, whichtoinvert=[1 if invert else 0]
        )
    except RuntimeError:
        return _apply_affine_to_vector_image(fixed, moving, affine_file, invert)


def _initial_extra_templates(extras, initial_extra_templates, weights, interpolator, primary, device):
    """Starting templates of the extra contrasts.

    These are their weighted averages, or the given images resampled onto the primary's grid.
    """
    if initial_extra_templates is None:
        return [_initial_template(contrast, weights, interpolator) for contrast in extras]
    if len(initial_extra_templates) != len(extras):
        raise ValueError(
            f"initial_extra_templates must hold one image per extra contrast ({len(extras)}), "
            f"got {len(initial_extra_templates)}"
        )
    target = _scalar_reference(primary)
    templates = []
    for c, image in enumerate(initial_extra_templates):
        if image.components != 1:
            raise ValueError(
                f"initial_extra_templates[{c}] must be a scalar image, got {image.components} components"
            )
        templates.append(_resample_vector_onto(image, target, device=device, interpolator=interpolator))
    return templates


def build_template(
    initial_template: Optional[ants.ANTsImage] = None,
    image_list: Optional[List[ants.ANTsImage]] = None,
    iterations: int = 3,
    gradient_step: float = 0.2,
    blending_weight: float = 0.75,
    weights: Optional[Sequence[float]] = None,
    use_no_rigid: bool = True,
    output_dir: Optional[str] = None,
    type_of_transform: str = "SyN",
    convergence_threshold: float = 0.0,
    affine_every_iteration: bool = False,
    extra_contrasts: Optional[Sequence[Sequence[ants.ANTsImage]]] = None,
    extra_contrast_weights: Optional[Sequence[float]] = None,
    initial_extra_templates: Optional[Sequence[ants.ANTsImage]] = None,
    initial_alignment: Optional[str] = None,
    initial_alignment_iterations: int = 1,
    initial_alignment_reference: Optional[Union[int, ants.ANTsImage, Sequence[ants.ANTsImage]]] = 0,
    output_interpolator: str = "linear",
    reorientation=None,
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build an unbiased group template with :func:`antstorch.syn.syn_registration`.

    Parameters
    ----------
    initial_template : ants.ANTsImage, optional
        Starting template.  Default: the weighted average of ``image_list``
        (resampled onto the first image).
    image_list : list of ants.ANTsImage
        The images (required, non-empty).
    iterations : int, default 3
    gradient_step : float in [0, 1], default 0.2
        Fraction of the mean warp applied to the template each iteration.
    blending_weight : float in (0, 1], default 0.75
        ``template = w * template + (1 - w) * sharpened(template)``;
        ``1`` disables sharpening.
    weights : sequence of float, optional
        Per-image weights (normalised to sum to 1; default uniform).
    use_no_rigid : bool, default True
        Average the affines without their rigid part
        (``ants.average_affine_transform_no_rigid``, falling back to
        ``ants.average_affine_transform`` if that fails).  Equivalent to
        ``useNoRigid`` in ``ants.build_template`` / ``syntx.build_template``.
    output_dir : str, optional
        Where per-iteration transforms are written (default: a temp directory).
    type_of_transform : str, default 'SyN'
        Per-image registration type, passed to ``syn_registration``.  With
        ``'SyN'`` the first iteration fits each image's affine; later
        iterations run ``'SyNOnly'`` starting from that cached affine (unless
        ``affine_every_iteration``), which avoids affine drift as the template
        moves.  Linear-only types (``'Rigid'``, ``'Affine'``, ...) are accepted:
        no mean warp exists then and only the affine is re-centred.
    convergence_threshold : float, default 0.0
        Stop early when the mean warp's RMS size falls below this (``0`` = never).
    affine_every_iteration : bool, default False
        ``True`` runs the full ``type_of_transform`` (affine included) every iteration.
    extra_contrasts : sequence of sequences of ants.ANTsImage, optional
        Further contrasts of the same subjects (``extra_contrasts[c][k]``
        belongs to ``image_list[k]``; scalar images such as T1-weighted
        volumes). Each is resampled onto its primary image's grid, gets its
        own template (initial average, the same shape update, scalar
        sharpening) and is registered together with the primary images
        through ``syn_registration``'s ``extra_contrasts``: a multi-contrast
        template as built by ``population_template`` with several input
        directories or ``antsMultivariateTemplateConstruction2.sh -k``.
    extra_contrast_weights : sequence of float, optional
        Weight of each extra contrast in the similarity (default 1 each; see
        ``syn_registration``).
    initial_extra_templates : sequence of ants.ANTsImage, optional
        Starting template of each extra contrast (one scalar image per entry
        of ``extra_contrasts``, resampled onto the primary template's grid):
        together with ``initial_template`` this continues a build from a
        previous result's ``template`` and ``extra_templates``.  Default: the
        weighted average of each contrast.
    initial_alignment : {'Translation', 'Rigid', 'Similarity', 'Affine'}, optional
        Linear pre-alignment of the inputs before the main loop, as
        ``antsMultivariateTemplateConstruction2.sh -r 1`` does with a rigid
        pass: every image is registered with this transform type (the extra
        contrasts included, see ``syn_registration``'s
        ``affine_extra_contrasts``) to ``initial_alignment_reference``, the
        aligned images are averaged, and that average starts the main loop
        instead of the average of the unaligned inputs.  ``'Affine'`` also
        places the template at the group's mean scale and shear.  Runs as
        linear-only builds under ``<output_dir>/initial_<type>`` with the same
        ``kwargs`` and no sharpening; skipped when ``initial_template`` is
        given.  Default: none.
    initial_alignment_iterations : int, default 1
        Passes of that pre-alignment (ANTs does one; two or three let an
        affine pass settle on the mean scale).  With a reference the first
        pass targets it and the later passes target the running average.
    initial_alignment_reference : int, ants.ANTsImage or sequence of ants.ANTsImage, optional
        Target of the first pre-alignment pass: an index into ``image_list``
        (default ``0``, the first input, with its extra contrasts), an image
        (with extra contrasts, a sequence: the primary reference followed by
        one image per extra contrast, e.g. an MNI template given as the
        ANTs script's ``-z``), or ``None`` for the plain average of the
        unaligned inputs (the ANTs script without ``-z``).  A single input
        avoids the ghosts that subsets sharing a pose leave in the average of
        unaligned inputs.  That pass averages the full transforms (rigid part
        included, whatever ``use_no_rigid`` says), so the template ends at
        the group's mean pose; a reference given as an image keeps its pose.
    output_interpolator : {'linear', 'cubic', 'bspline', 'nearest'}, default 'linear'
        Interpolation of the warped images (see ``syn_registration``) and of
        the template in the shape update; ``'cubic'`` preserves the inputs'
        sharpness through repeated resampling.
    reorientation : optional
        How the channels of multi-component images transform (see
        ``syn_registration``); also applied to the template in the shape
        update.  :func:`antstorch.syn.build_fod_template` passes a
        :class:`antstorch.syn.fod.FODReorientation`.
    verbose : bool, default False
    **kwargs
        Forwarded to ``syn_registration`` (``regularizer``, ``syn_metric``,
        ``levels``, ``reg_iterations``, ``device``, ...).  ``initial_affine`` and
        ``outprefix`` are managed here and rejected.

    Returns
    -------
    dict
        ``template``; ``warped_images`` / ``fwdtransforms`` / ``invtransforms``
        (per image, last iteration); ``extra_templates`` and
        ``warped_extra_images`` (per extra contrast; the latter per image,
        last iteration); ``convergence`` (mean absolute template
        change per iteration); ``shape_residuals`` (per iteration: ``l2_norm``,
        ``membrane_energy``, ``bending_energy`` of the mean warp);
        ``n_iterations``; ``weights``; ``work_dir``; ``initial_alignment``
        (the pre-alignment templates, pass results and reference, or
        ``None``); ``elapsed_sec``.
    """
    t0 = time.time()
    if image_list is None or len(image_list) == 0:
        raise ValueError("image_list must be a non-empty list of ANTsImages.")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if not (0.0 < blending_weight <= 1.0):
        raise ValueError(f"blending_weight must be in (0, 1], got {blending_weight}")
    if not (0.0 <= gradient_step <= 1.0):
        raise ValueError(f"gradient_step must be in [0, 1], got {gradient_step}")
    for name in _MANAGED_KWARGS:
        if name in kwargs:
            raise TypeError(f"build_template manages {name!r} itself; remove it from the keyword arguments")

    n = len(image_list)
    weights = _normalize_weights(weights, n)
    if output_interpolator not in _OUTPUT_INTERPOLATORS:
        raise ValueError(f"output_interpolator must be one of {_OUTPUT_INTERPOLATORS}, got {output_interpolator!r}")
    interpolator = output_interpolator
    torch_device = _resolve_device(kwargs.get("device"))
    components = image_list[0].components
    if any(image.components != components for image in image_list):
        raise ValueError("all images in image_list must have the same number of components")
    # The ANTs path (ants.apply_transforms) resamples a scalar template linearly; multi-component
    # templates and other interpolators go through torch.
    torch_template_update = components > 1 or interpolator != "linear"

    extras = _prepare_template_extras(image_list, extra_contrasts, interpolator, torch_device)
    n_extra = len(extras)
    if extra_contrast_weights is not None and len(extra_contrast_weights) != n_extra:
        raise ValueError(
            f"extra_contrast_weights must have one entry per extra contrast ({n_extra}), "
            f"got {len(extra_contrast_weights)}"
        )

    work_dir = tempfile.mkdtemp(prefix="antstorch_tmpl_") if output_dir is None else output_dir
    os.makedirs(work_dir, exist_ok=True)

    # Linear pre-alignment (ANTs' -r 1): linear-only builds of their own, whose templates start the main loop.
    initial_alignment_result: Optional[Dict[str, Any]] = None
    if initial_alignment is not None and initial_template is None:
        if initial_alignment not in _LINEAR_TRANSFORM_TYPES:
            raise ValueError(f"initial_alignment must be one of {_LINEAR_TRANSFORM_TYPES}, got {initial_alignment!r}")
        passes = int(initial_alignment_iterations)
        if passes < 1:
            raise ValueError(f"initial_alignment_iterations must be at least 1, got {initial_alignment_iterations}")
        reference, reference_extras, reference_name, reference_is_input = _initial_alignment_reference(
            initial_alignment_reference, image_list, extra_contrasts
        )
        if verbose:
            print(
                f"[build_template] Initial {initial_alignment} alignment, {passes} pass(es), "
                f"first against {reference_name}"
            )
        sub = dict(
            image_list=image_list,
            weights=weights,
            type_of_transform=initial_alignment,
            extra_contrasts=extra_contrasts,
            extra_contrast_weights=extra_contrast_weights,
            blending_weight=1.0,
            output_interpolator=output_interpolator,
            reorientation=reorientation,
            verbose=verbose,
            **kwargs,
        )
        base = os.path.join(work_dir, f"initial_{initial_alignment.lower()}")
        pass_results: List[Dict[str, Any]] = []
        if reference is not None:
            # First pass against one image, not the average of the unaligned inputs (see
            # initial_alignment_reference); with an input as the reference the full transforms are
            # averaged, rigid part included, so the template lands at the group's mean pose.
            pass_results.append(
                build_template(
                    iterations=1, output_dir=base + "_reference", initial_template=reference,
                    initial_extra_templates=reference_extras,
                    use_no_rigid=use_no_rigid and not reference_is_input, **sub,
                )
            )
            passes -= 1
        if passes > 0:
            seed = pass_results[-1] if pass_results else None
            pass_results.append(
                build_template(
                    iterations=passes, output_dir=base, use_no_rigid=use_no_rigid,
                    initial_template=None if seed is None else seed["template"],
                    initial_extra_templates=None if seed is None else seed["extra_templates"], **sub,
                )
            )
        initial_template = pass_results[-1]["template"]
        initial_extra_templates = pass_results[-1]["extra_templates"]
        initial_alignment_result = {
            "template": initial_template,
            "extra_templates": initial_extra_templates,
            "passes": pass_results,
            "reference": reference_name,
        }

    def make_outprefix(it: int, k: int) -> str:
        d = os.path.join(work_dir, f"iter{it:02d}_img{k:04d}")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "out")

    if initial_template is None:
        xavg = _initial_template(image_list, weights, interpolator)
    else:
        xavg = initial_template.clone()
    xavg_extras = _initial_extra_templates(
        extras, initial_extra_templates, weights, interpolator, xavg, torch_device
    )

    shape_residuals: List[Dict[str, float]] = []
    convergence: List[float] = []
    last_warped: List[Optional[ants.ANTsImage]] = [None] * n
    last_fwd: List[List[str]] = [[] for _ in range(n)]
    last_inv: List[List[str]] = [[] for _ in range(n)]
    last_warped_extras: List[List[Optional[ants.ANTsImage]]] = [[None] * n for _ in range(n_extra)]
    # Each image's affine from its last full registration, as an ITK-order
    # (matrix, translation) pair, fed back as ``initial_affine`` to 'SyNOnly'.
    image_affine: List[Optional[Tuple[Any, Any]]] = [None] * n

    is_syn = type_of_transform.lower() == "syn"
    for it in range(iterations):
        if verbose:
            print(f"[build_template] Iteration {it + 1}/{iterations}")
        iter_tot = "SyNOnly" if (is_syn and it > 0 and not affine_every_iteration) else type_of_transform
        affine_files: List[str] = []
        wavg: Optional[ants.ANTsImage] = None
        xavg_new: Optional[ants.ANTsImage] = None
        xavg_extras_new: List[Optional[ants.ANTsImage]] = [None] * n_extra
        has_field = False

        for k in range(n):
            if verbose:
                print(f"  Registering subject {k + 1}/{n} ...", end=" ", flush=True)
            reg_kw = dict(kwargs)
            if iter_tot == "SyNOnly" and image_affine[k] is not None:
                reg_kw["initial_affine"] = image_affine[k]
            if n_extra:
                reg_kw["extra_contrasts"] = [(xavg_extras[c], extras[c][k]) for c in range(n_extra)]
                reg_kw["extra_contrast_weights"] = extra_contrast_weights
            w1 = syn_registration(
                xavg,
                image_list[k],
                type_of_transform=iter_tot,
                outprefix=make_outprefix(it, k),
                output_interpolator=output_interpolator,
                reorientation=reorientation,
                verbose=verbose,
                **reg_kw,
            )
            has_field = w1["jacobian"] is not None  # dense stage ran
            # fwdtransforms = [warp, affine] (SyN), [warp] (SyNOnly, no affine) or [affine] (linear only)
            # An affine passed in as ``initial_affine`` (cached from iteration 0) is not new
            # information: the template was already recentred by its average, so averaging
            # it again every iteration would compound the same scale/shear change on the
            # template (observed: identical avgAffine_k files, 4 iterations, ~10% scale drift).
            # From iteration 1 on only the dense warp therefore recentres the template, as in
            # syntx (where the mean affine is the identity after iteration 0).
            if w1["fwdtransforms"][-1].endswith(".mat") and "initial_affine" not in reg_kw:
                affine_files.append(w1["fwdtransforms"][-1])
            if iter_tot != "SyNOnly" and w1["affine_matrix"] is not None:
                image_affine[k] = (w1["affine_matrix"], w1["affine_translation"])

            last_warped[k], last_fwd[k], last_inv[k] = w1["warpedmovout"], w1["fwdtransforms"], w1["invtransforms"]

            weight = float(weights[k])
            warped = w1["warpedmovout"] * weight
            xavg_new = warped if xavg_new is None else xavg_new + warped
            for c in range(n_extra):
                last_warped_extras[c][k] = w1["warpedmovout_extras"][c]
                warped_extra = w1["warpedmovout_extras"][c] * weight
                if xavg_extras_new[c] is None:
                    xavg_extras_new[c] = warped_extra
                else:
                    xavg_extras_new[c] = xavg_extras_new[c] + warped_extra
            if has_field:
                field = ants.image_read(w1["fwdtransforms"][0]) * weight
                wavg = field if wavg is None else wavg + field
            if verbose:
                print("done")

        if has_field and wavg is not None:
            sr = _compute_shape_residual(wavg)
        else:
            sr = {"l2_norm": 0.0, "membrane_energy": 0.0, "bending_energy": 0.0}
        shape_residuals.append(sr)
        if verbose:
            print(
                f"  Iteration {it} shape residual - L2: {sr['l2_norm']:.4f}, "
                f"membrane: {sr['membrane_energy']:.6f}, bending: {sr['bending_energy']:.8f}"
            )

        xavg_pre = xavg.clone()

        affine_fn: Optional[str] = None
        if affine_files:
            try:
                if use_no_rigid:
                    avg_affine = ants.average_affine_transform_no_rigid(affine_files)
                else:
                    avg_affine = ants.average_affine_transform(affine_files)
            except Exception:
                if not use_no_rigid:
                    raise
                avg_affine = ants.average_affine_transform(affine_files)
            affine_fn = os.path.join(work_dir, f"avgAffine_{it}.mat")
            ants.write_transform(avg_affine, affine_fn)

        # antsApplyTransforms only needs the reference geometry from ``fixed``; a scalar
        # channel keeps it away from the multi-component template.
        reference = _scalar_reference(xavg_new)
        scaled: Optional[ants.ANTsImage] = None
        warp_fn: Optional[str] = None
        if has_field and wavg is not None:
            scaled = wavg * ((-1.0) * gradient_step)
            if affine_fn is not None:
                scaled = _apply_affine_to_vectors(reference, scaled, affine_fn, invert=True)
            warp_fn = os.path.join(work_dir, f"avgWarp_{it}.nii.gz")
            ants.image_write(scaled, warp_fn)
        # The extra-contrast templates follow the primary template through the same update.
        update = dict(
            scaled_warp=scaled, warp_fn=warp_fn, affine_fn=affine_fn, use_torch=torch_template_update,
            interpolator=interpolator, device=torch_device,
        )
        xavg = _apply_shape_update(xavg_new, reorientation=reorientation, **update)
        xavg_extras = [_apply_shape_update(image, **update) for image in xavg_extras_new]

        if blending_weight < 1.0:
            xavg = _sharpen_image(xavg, blending_weight)
            xavg_extras = [_sharpen_image(image, blending_weight) for image in xavg_extras]

        change = _template_change(xavg_pre, xavg)
        convergence.append(change)
        if verbose:
            print(f"  Template mean absolute change: {change:.6f}")

        if convergence_threshold > 0.0 and sr["l2_norm"] < convergence_threshold:
            if verbose:
                print(f"  Converged: shape residual {sr['l2_norm']:.6f} < {convergence_threshold:.6f}")
            break

    return {
        "template": xavg,
        "warped_images": last_warped,
        "extra_templates": xavg_extras,
        "warped_extra_images": last_warped_extras,
        "fwdtransforms": last_fwd,
        "invtransforms": last_inv,
        "convergence": convergence,
        "shape_residuals": shape_residuals,
        "n_iterations": len(shape_residuals),
        "weights": weights,
        "work_dir": work_dir,
        "initial_alignment": initial_alignment_result,
        "elapsed_sec": time.time() - t0,
    }
