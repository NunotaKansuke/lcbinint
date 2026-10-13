# Bounded hot-path trials

1. Certified quartic count reused for discriminant sign in transport. Accepted; no second topology screen needed. Default callers without a certified node count unchanged.
2. Transport corrector step tolerance1e-10 instead1e-13. Normal192 tolerance comparisons passed, but median-per-case averaged time did not improve (0.30292 vs0.29771 ms count-reuse trial); default change reverted. Raw transport_tolerance.tsv and saved binary retained as rejected trial.
3. Complete simple-root set from certified count permits alternating inside/outside signs. Widest gap sampled once; nonfinite/zero sign falls back to full scan. HOLO_ARC_SIGN_AUDIT checks every interval against the old map evaluation. Accepted after all896 audited conditions passed.
4. Frontier count probes compute quartic coefficients directly in long double, avoiding construction of unused radial derivatives and angular jet multiplication/division. Same polynomial and precision, possible last-bit rounding changes. Accepted: boundary preparation reduced12.2–12.5%, node counts unchanged and maximum relative magnification change1.1945e-13 over896 conditions.

All changes remain in the existing experimental checkout. Source baseline is the completed v15 pipeline, current master seed library.
