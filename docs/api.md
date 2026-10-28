# API guide

Application and architecture entry points are exported from `antstorch`.
Registration-specific APIs can also be imported from their subpackages.

| Entry point | Input | Result |
| --- | --- | --- |
| `antstorch.create_unet_model_2d` | Channel count, output count, architecture options | A PyTorch module; inputs use NCHW order |
| `antstorch.create_unet_model_3d` | Channel count, output count, architecture options | A PyTorch module; inputs use NCDHW order |
| `antstorch.brain_extraction` | ANTsImage and modality | Probability image for `t1`; other modalities can return different structures |
| `antstorch.deep_atropos` | T1 image and preprocessing options | Dictionary containing segmentation and probability images |
| `antstorch.syn.syn_registration` | Fixed and moving ANTsImages, scalar or multi-component, optional `extra_contrasts` and `reorientation` | Warped images (`output_interpolator`), warped extra contrasts and file-based transform lists |
| `antstorch.syn.build_template` | List of ANTsImages, optional `extra_contrasts`, `initial_template`, `initial_alignment` (the ANTs script's `-r 1`) | Unbiased template, one per extra contrast, per-image warped images and transforms |
| `antstorch.syn.fod_registration` / `build_fod_template` | MRtrix3 spherical harmonic FOD images, `lmax`, `directions`, `modulate`, plus the SyN or template options | The same results, with the FODs reoriented |
| `antstorch.syn.fod_image_read` / `fod_image_write` | MRtrix3 spherical harmonic images as 4-D NIfTI; `.mif`, `.odx` and DSI Studio files with the `odx` package | Multi-component ANTsImage, and back |
| `antstorch.syn.sh.FODReorienter` | SH image and per-voxel direction maps | Reoriented SH image (apodized point spread function method) |
| `antstorch.bspline_flows.n4_bias_field_correction_tensor` | Image tensor, domain, optional mask | Corrected image tensor or bias field |
| `antstorch.bspline_flows.bspline_svf_registration` | Image tensors and physical domains | Warped image, displacement fields and optimization history |

## Image geometry

ANTs arrays use `(X, Y[, Z])`; PyTorch tensors use `(N, C, [Z,] Y, X)`.
Reversing spatial axes does not reverse the physical vector components. Preserve
spacing, origin and direction when converting images. The
[B-spline tutorial](antsx_tutorial_bspline_flows.md) explains these conversions.

## Transforms

SyN returns paths compatible with `ants.apply_transforms`. B-spline SVF
registration returns tensors, with any initial affine returned separately.
With extra contrasts, SyN's internal affine fit uses every contrast as a channel
(`affine_extra_contrasts`, default on), as the ANTs multivariate template script does.
Use label interpolation when resampling segmentations. See the
[SyN guide](antsx_tutorial_syn.md) for file ordering and inverse transforms.

## Full parameter documentation

In an installed environment, use `help(antstorch.brain_extraction)` or
`help(antstorch.syn.syn_registration)` for the current implementation's arguments.
The [source repository](https://github.com/ANTsX/ANTsTorch) contains the full
implementations. This documentation build intentionally does not import the
scientific stack or execute model inference.
