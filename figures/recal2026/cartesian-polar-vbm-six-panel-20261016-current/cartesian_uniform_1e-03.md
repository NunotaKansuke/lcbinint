# Adaptive Cartesian six-panel comparison

Profile: `uniform`; target: `0.001`; conditions: `3608`.

The upper panels show cell median `t_VBM/t_method`; values above one favor this adaptive flood-fill route. The lower panels show cell p95 absolute relative difference from the adjudicated saved RelTol=1e-6 reference, not difference from VBM.

Finite timing/value rows: `3456`; estimate accepted: `3258`; actual reference pass: `3443/3608`; false accepts: `12`.
Median VBM/method ratio over finite rows: `0.235559`; among actual reference-pass rows: `0.235622`.
Reference-error p95: `0.000475699`; maximum: `0.00172647`.

Agreement is empirical, not a formal error certificate. Per-condition values and rejections are in `joined.tsv`.
