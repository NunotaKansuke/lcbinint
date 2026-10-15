# Cartesian endpoint/startup experiments (2026-10-15)

- `selected-v7-*`: fully audited square 16x16 candidate; 61 strict false accepts versus 408 before this work. Native archived in `frozen-v7/` and private source in `candidate-v7-source/`.
- `rectangular-v8-*`: INVALID rectangle experiment. Returned area used hx*hy while the legacy wrapper multiplies hx^2; values were halved. Not a speed/accuracy result.
- `rectangular-v9-*`: normalization repaired, component flux ledger added. Independent heldout216: 0 false accepts,72 exceptions,30 rejections. Paired128 pilot: uniform1.296/1.153, LD1.575/1.278 median speed ratios, with all failures preserved. Not a full accuracy audit.
- `rectangular-v10-*`: trial physical-phase quartic squared-width fit from four existing rows; fifth/sixth validate. Unit10000 synthetic models pass,506 moments vs55digit mpmath pass (7.34e-16 scaled error). Full14432: false26/125 uniform,16/112LD; exceptions78/86 per tolerance. The many new1e-4 failures show this is not a sufficiently validated replacement for square startup. Source/native archived as candidate-v10-source/frozen-v10. The extra model degree was removed from the next candidate; it gave little improvement in mapped rays and added a new coarse false acceptance.
- v11: return to v7 endpoint order, retain corrected rectangle normalization, apply existing coarse-start guards and owner closure by actual row/column dimensions. Verify before selecting any default.
- `independent_rows.py`: SciPy full-row integration emitted warnings and can miss branches; its output is NOT an independent acceptance reference.

No global tolerance tightening or case-ID rescue was added for case1604 at1.088e-4. Scientific failures, unmet requests, exceptions and discarded trials remain in their original output directories.

- v11 targeted14 inputs x2 targets:0 false accepts,0exceptions,8unmet requests. Correcting coarse-row guard did not make every request accepted. Independent heldout216 has0false,72exceptions,30unmet;144internal checks passed.
- v12 limits new16x8 default to initial_bins16; explicit8/15/32 defaults keep square grids. No further endpoint order or error safety factor changes.

- v12 full paired256: casewise median speed ratios vs before-work baseline Uniform1.1502/1.0492, LD1.0612/1.0187. Polar unchanged controls1.0045-1.0074. The rectangle gain does not survive the actual-row correctness guard for LD1e-4; fully audited squarev7 gaveLD1.0544/1.0592 with fewer false accepts. **Reject rectangular default and restore exact v7 private source.** This decision is based on whole-cohort speed/accuracy, not case1604's tiny excess. v12 source/native, experimental drivers, failures and timing remain archived.
