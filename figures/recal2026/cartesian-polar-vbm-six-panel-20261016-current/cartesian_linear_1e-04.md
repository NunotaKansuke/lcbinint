# Adaptive Cartesian six-panel comparison

Profile: `linear`; target: `0.0001`; conditions: `3608`.

The upper panels show cell median `t_VBM/t_method`; values above one favor this adaptive flood-fill route. The lower panels show cell p95 absolute relative difference from the adjudicated saved RelTol=1e-6 reference, not difference from VBM.

Finite timing/value rows: `3448`; estimate accepted: `3211`; actual reference pass: `3413/3608`; false accepts: `18`.
Median VBM/method ratio over finite rows: `3.52909`; among actual reference-pass rows: `3.57443`.
Reference-error p95: `6.51455e-05`; maximum: `0.00110429`.

Agreement is empirical, not a formal error certificate. Per-condition values and rejections are in `joined.tsv`.
