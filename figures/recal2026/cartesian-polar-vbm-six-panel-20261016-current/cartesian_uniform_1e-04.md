# Adaptive Cartesian six-panel comparison

Profile: `uniform`; target: `0.0001`; conditions: `3608`.

The upper panels show cell median `t_VBM/t_method`; values above one favor this adaptive flood-fill route. The lower panels show cell p95 absolute relative difference from the adjudicated saved RelTol=1e-6 reference, not difference from VBM.

Finite timing/value rows: `3456`; estimate accepted: `3202`; actual reference pass: `3406/3608`; false accepts: `26`.
Median VBM/method ratio over finite rows: `0.300116`; among actual reference-pass rows: `0.30901`.
Reference-error p95: `7.41874e-05`; maximum: `0.00162351`.

Agreement is empirical, not a formal error certificate. Per-condition values and rejections are in `joined.tsv`.
