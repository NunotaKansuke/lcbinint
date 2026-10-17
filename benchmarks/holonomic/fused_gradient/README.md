# Archived fused-gradient experiment

This directory contains historical notebooks and measurements for the seeded
fused-gradient implementation. Its separate Python entry point has been
removed. The compiled kernel is built as a private dependency and called by the
unified `LightCurve(..., with_gradient=True)` API for supported eager
static-binary calls. The notebooks and outputs remain historical evidence.

Use one `LightCurve` call for values and optional gradients:

```python
curve = lc.LightCurve(options=lc.Options(param_type="vbm"))
values = curve(times, params, with_gradient=False)
with_gradient = curve(times, params, with_gradient=True)
gradient = with_gradient.gradients
```

For a supported static single-source binary, the branch default selects the
algebraic-boundary backend. Its shared selector retains the point-source and
hexadecapole fast paths and uses the image-boundary integral when needed.
Models outside that backend's supported scope retain the standard router.

The historical results in this directory measure the fused kernel itself and
do not certify its gradients. The current public-API route and its value/status
differences from value-only algebraic-boundary calls are documented in the
library-comparison evidence.
