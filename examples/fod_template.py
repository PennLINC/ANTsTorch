"""Build an unbiased FOD template from spherical harmonic images with reorientation.

The inputs are MRtrix3-convention SH images stored as 4-D NIfTI files (convert
``.mif`` files with ``mrconvert``). Each iteration registers every image to the
current template with FOD reorientation, averages the warped images per
coefficient, moves the template by a fraction of the mean warp (with
reorientation) and sharpens it per coefficient. Warped images and the template
are resampled with an interpolating cubic kernel by default, like MRtrix3's
``mrtransform``, so the template keeps the angular contrast of its inputs.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="FOD images (4-D NIfTI, MRtrix3 SH convention)")
    parser.add_argument("--output-dir", type=Path, default=Path("example_outputs/fod_template"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--gradient-step", type=float, default=0.2)
    parser.add_argument("--blending-weight", type=float, default=0.75, help="1 disables sharpening")
    parser.add_argument("--metric", default="lncc_joint")
    parser.add_argument("--flow-sigma", type=float, default=2.0, help="update field smoothing (voxels)")
    parser.add_argument("--total-sigma", type=float, default=2.0, help="total field smoothing (mm; 0 disables)")
    parser.add_argument("--reg-iterations", type=int, nargs="+", default=(50, 50, 50))
    parser.add_argument("--output-interpolator", default="cubic", choices=("linear", "cubic", "bspline", "nearest"),
                        help="interpolation of the warped images and of the template in the shape update")
    args = parser.parse_args()
    from antstorch.syn import build_template
    from antstorch.syn.fod import fod_image_read, fod_image_write

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = [fod_image_read(path) for path in args.images]
    result = build_template(
        image_list=images, iterations=args.iterations, gradient_step=args.gradient_step,
        blending_weight=args.blending_weight, output_dir=str(args.output_dir / "work"), type_of_transform="SyN",
        fod=True, syn_metric=args.metric, levels=(4, 2, 1), reg_iterations=tuple(args.reg_iterations),
        flow_sigma=args.flow_sigma, total_sigma=args.total_sigma, output_interpolator=args.output_interpolator,
        device=args.device, verbose=True,
    )
    fod_image_write(result["template"], str(args.output_dir / "template.nii.gz"))
    for path, warped in zip(args.images, result["warped_images"]):
        fod_image_write(warped, str(args.output_dir / f"warped_{Path(path).name}"))
    print("Mean warp RMS per iteration (mm):", [round(r["l2_norm"], 4) for r in result["shape_residuals"]])
    print("Template mean absolute change per iteration:", [round(c, 6) for c in result["convergence"]])


if __name__ == "__main__":
    main()
