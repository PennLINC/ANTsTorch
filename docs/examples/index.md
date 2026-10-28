# Runnable examples

Install ANTsTorch first, then run these commands from the repository root.
The scripts complement the [ANTsX tutorial](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621);
they use ANTsTorch rather than assuming ANTsPyNet APIs are interchangeable.

| Example | Command | Data and outputs |
| --- | --- | --- |
| U-Net forward pass | `python examples/unet_2d.py` | Offline; prints `(1, 3, 64, 64)` probabilities from random weights |
| N4 correction | `python examples/n4_synthetic.py` | Offline; prints corrected tensor shape; short synthetic demonstration |
| SyN registration | `python examples/syn_registration.py` | Downloads r16/r64; writes warped image, Jacobian and transforms to `example_outputs/syn/` |
| FOD registration | `python examples/fod_registration.py fixed_fod.nii.gz moving_fod.nii.gz` | Provide two MRtrix3 SH images (4-D NIfTI); `--extra` adds a T1w pair; writes the reoriented warped FOD image, Jacobian and transforms to `example_outputs/fod/` |
| FOD template | `python examples/fod_template.py sub-*_fod.nii.gz` | Provide several MRtrix3 SH images; `--extra` adds one T1w per subject; writes the templates to `example_outputs/fod_template/` |
| Brain extraction | `python examples/brain_extraction.py /path/to/t1.nii.gz` | Downloads pretrained assets; writes probability, thresholded mask and brain image to `example_outputs/brain/` |

The registration, template and brain-extraction scripts accept `--device` and `--output-dir`; `--help` lists their
options. Existing files with the same output names are overwritten. Brain
extraction uses a 0.5 probability threshold; inspect the mask on your input image.
Scripts do not open plot windows and are not executed during documentation builds.

```{toctree}
:maxdepth: 1

unet_2d
n4_synthetic
syn_registration
fod_registration
fod_template
brain_extraction
```
