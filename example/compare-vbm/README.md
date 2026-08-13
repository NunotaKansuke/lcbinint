# lcbinint / VBM comparison example

This example shows the public `LightCurve` callable API and a light-curve-level
speed/accuracy comparison:

- create one reusable callable with numerical options and limb darkening fixed
- evaluate finite-source light curves
- compare it against VBM when `VBMicrolensing` is installed
- plot the VBM light curve as a line and lcbinint as larger scatter points
- plot the epoch-by-epoch relative error with respect to VBM

Run the script version:

```bash
PYTHONPATH=build python example/compare-vbm/quickstart_compare_vbm.py
PYTHONPATH=build python example/compare-vbm/quickstart_compare_vbm_triple.py
```

The binary script currently does not call the optional lcbinint warm-up plan:
the first ordinary full-light-curve call is included in the timing samples for
both engines. Both engines use the same 400 epochs and the same requested
relative tolerance (`1e-3`). The script prints median timing distributions and
accuracy summaries, then writes a comparison plot beside the script. The
notebook versions contain the same binary- and triple-lens workflows in
interactive form.
