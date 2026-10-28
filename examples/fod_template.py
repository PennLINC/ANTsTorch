"""Build an unbiased template from MRtrix3 FOD images (spherical harmonics, 4-D NIfTI)."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--extra", nargs="+", default=[], metavar="IMAGE", help="one per FOD image")
    parser.add_argument("--initial-alignment", choices=("Rigid", "Affine"), default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("example_outputs/fod_template"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import ants
    from antstorch.syn import build_fod_template, fod_image_read, fod_image_write

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extras = [ants.image_read(path) for path in args.extra]
    result = build_fod_template(
        image_list=[fod_image_read(path) for path in args.images], iterations=3,
        output_dir=str(args.output_dir / "work"), initial_alignment=args.initial_alignment,
        extra_contrasts=[extras] if extras else None, levels=(4, 2, 1),
        reg_iterations=(50, 50, 50), total_sigma=1.0, affine_every_iteration=True,
        device=args.device, verbose=True,
    )
    fod_image_write(result["template"], str(args.output_dir / "template.nii.gz"))
    for c, template in enumerate(result["extra_templates"], start=1):
        ants.image_write(template, str(args.output_dir / f"template_extra{c}.nii.gz"))
    rms = [round(r["l2_norm"], 3) for r in result["shape_residuals"]]
    print("Mean warp RMS per iteration (mm):", rms)


if __name__ == "__main__":
    main()
