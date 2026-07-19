# Triple finite-source auto calibration artifact

`calibrated-rules.json` freezes the triple-lens `nbin="auto"` quantile
regression used by the runtime selector.  It was generated with
`tests/diagnostics/triple_finite_source_calibration.py` using separate
discovery and holdout seeds.  Fixed Cartesian convergence sequences establish
the labels; VBMicrolensing is not used as an oracle.

The frozen target is absolute tolerance `1e-4` plus relative tolerance `1e-3`.
The run retained 137 uncensored training labels and 146 independent holdout
labels across two tuning and one final validation split.  Two residuals were
both in the source/caustic contact band, so the frozen rule includes a 100-bin
floor for `d_caustic/rho <= 0.5`; it removes all observed under-predictions.
Tighter requested relative tolerances use the runtime's documented square-root
conservative scaling.
