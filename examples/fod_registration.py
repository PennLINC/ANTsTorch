"""Register two MRtrix3 FOD images (spherical harmonics as 4-D NIfTI) with reorientation."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixed")
    parser.add_argument("moving")
    parser.add_argument("--extra", nargs=2, action="append", default=[], metavar=("FIX", "MOV"))
    parser.add_argument("--output-dir", type=Path, default=Path("example_outputs/fod"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import ants
    from antstorch.syn import fod_image_read, fod_image_write, fod_registration

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extras = [(ants.image_read(f), ants.image_read(m)) for f, m in args.extra]
    result = fod_registration(
        fod_image_read(args.fixed), fod_image_read(args.moving), type_of_transform="SyN",
        levels=(4, 2, 1), reg_iterations=(100, 70, 50), extra_contrasts=extras or None,
        device=args.device, outprefix=str(args.output_dir / "moving_to_fixed_"), verbose=True,
    )
    fod_image_write(result["warpedmovout"], str(args.output_dir / "warped.nii.gz"))
    for c, warped in enumerate(result["warpedmovout_extras"], start=1):
        ants.image_write(warped, str(args.output_dir / f"warped_extra{c}.nii.gz"))
    ants.image_write(result["jacobian"], str(args.output_dir / "jacobian.nii.gz"))
    print("Forward transforms:", result["fwdtransforms"])


if __name__ == "__main__":
    main()
