# Kepler LOM reference epoch

This example shows the Keplerian orbital-motion path through the public
`LightCurve` API.

`VBMicrolensing.BinaryLightCurveKepler` uses the event `t0` as the orbital
reference epoch.  `lcbinint` keeps the more explicit model convention:
`LightCurve(t_ref=...)` sets the fixed reference epoch for both parallax and
lens orbital motion.  Setting `t_ref=t0` gives the VBMicrolensing-compatible
case; setting a fixed epoch keeps the orbital phase anchored while fitted
parameters such as `t0` vary.

Run:

```bash
PYTHONPATH=build python example/kepler-lom/kepler_lom_reference_time.py
```
