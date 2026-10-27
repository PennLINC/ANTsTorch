"""Register two fiber orientation distribution (FOD) images with reorientation.

The inputs are spherical harmonic images in the MRtrix3 convention (e.g. the
output of ``dwi2fod`` or ``mtnormalise``) stored as 4-D NIfTI files; convert
``.mif`` files first with ``mrconvert fod.mif fod.nii.gz``. The warped moving
image is written as a 4-D NIfTI that MRtrix3 reads as an SH image.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixed", help="fixed FOD image (4-D NIfTI, MRtrix3 SH convention)")
    parser.add_argument("moving", help="moving FOD image (4-D NIfTI, MRtrix3 SH convention)")
    parser.add_argument("--output-dir", type=Path, default=Path("example_outputs/fod"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metric", default="lncc_joint", help="'lncc_joint' (CC over the coefficient vector) or 'mse' (as mrregister)")
    parser.add_argument("--type", default="SyN", dest="type_of_transform", help="'SyN' (affine on the l=0 term, then SyN) or 'SyNOnly'")
    parser.add_argument("--lmax", type=int, nargs="+", default=None, help="lmax per pyramid level (default 0 2 4)")
    parser.add_argument("--flow-sigma", type=float, default=3.0, help="update field smoothing (voxels)")
    parser.add_argument("--total-sigma", type=float, default=0.0, help="total field smoothing (mm; 0 disables)")
    parser.add_argument("--iterations", type=int, nargs="+", default=(100, 70, 50))
    parser.add_argument("--modulate", action="store_true", help="apply FOD modulation to the warped output")
    parser.add_argument("--output-interpolator", default="cubic", choices=("linear", "cubic", "bspline", "nearest"),
                        help="interpolation of the warped output (cubic: interpolating Catmull-Rom, as mrtransform)")
    args = parser.parse_args()
    import ants
    from antstorch.syn import syn_registration
    from antstorch.syn.fod import fod_image_read, fod_image_write

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = fod_image_read(args.fixed)
    moving = fod_image_read(args.moving)
    result = syn_registration(
        fixed, moving, type_of_transform=args.type_of_transform, fod=True, fod_lmax=args.lmax,
        fod_modulate=args.modulate, syn_metric=args.metric, levels=(4, 2, 1), reg_iterations=tuple(args.iterations),
        flow_sigma=args.flow_sigma, total_sigma=args.total_sigma, neighborhood_radius=4, grad_step=0.2,
        output_interpolator=args.output_interpolator, device=args.device,
        outprefix=str(args.output_dir / "moving_to_fixed_"), verbose=True,
    )
    fod_image_write(result["warpedmovout"], str(args.output_dir / "moving_in_fixed.nii.gz"))
    ants.image_write(result["jacobian"], str(args.output_dir / "jacobian.nii.gz"))
    print("Forward transforms (ANTs convention):", result["fwdtransforms"])
    print("Per-level lmax:", result["provenance"]["fod_lmax"], "metric:", result["provenance"]["syn_metric"])


if __name__ == "__main__":
    main()
