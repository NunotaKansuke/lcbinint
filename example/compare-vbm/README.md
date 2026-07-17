# lcbinint / VBM comparison example

This example shows the public `LightCurve` callable API:

- create one reusable callable with numerical options and limb darkening fixed
- evaluate finite-source light curves
- compare it against VBM when `VBMicrolensing` is installed
- plot the light curve, relative residual, source trajectory, and caustic

Run the script version:

```bash
PYTHONPATH=build python example/compare-vbm/quickstart_compare_vbm.py
PYTHONPATH=build python example/compare-vbm/quickstart_compare_vbm_triple.py
```

The scripts print median timing distributions and accuracy summaries, then
write comparison plots beside the scripts. The notebook versions contain the
same binary- and triple-lens workflows in interactive form.
