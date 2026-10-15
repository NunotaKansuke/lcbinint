# Controlled pure-kernel speed/diagnostic map

Profile: `linear`
Target: `0.001`
Measured epoch points: `3608`
Input parts: `3` (read serially in numeric order)
Minimum cell population: `8`
Gray cells are masked: the upper map has fewer than `8` finite timing ratios; the lower map has fewer than `8` finite cross-code errors.

The top row is the cell median of `R=t_VBM/t_lcbinint`; `R>1` favours lcbinint.
The bottom row is the cell p95 of the stored cross-code relative difference against VBM. It is a disagreement diagnostic, not an independent accuracy certificate.

| statistic | value |
|---|---:|
| lcbinint win rate (`R>1`) | `53.437%` |
| median `R` | `1.08521` |
| p10 `R` | `0.27722` |
| p90 `R` | `4.25765` |
| cross-code error records | `3608` |
| cross-code error p95 | `0.000265138` |
