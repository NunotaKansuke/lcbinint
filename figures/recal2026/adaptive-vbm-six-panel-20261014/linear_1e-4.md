# Controlled pure-kernel speed/diagnostic map

Profile: `linear`
Target: `0.0001`
Measured epoch points: `3608`
Input parts: `1` (read serially in numeric order)
Minimum cell population: `8`

The top row is the cell median of `R=t_VBM/t_lcbinint`; `R>1` favours lcbinint.
The bottom row is the cell p95 of the stored cross-code relative difference against VBM. It is a disagreement diagnostic, not an independent accuracy certificate.

| statistic | value |
|---|---:|
| lcbinint win rate (`R>1`) | `70.505%` |
| median `R` | `2.36537` |
| p10 `R` | `0.192672` |
| p90 `R` | `11.6577` |
| cross-code error records | `3448` |
| cross-code error p95 | `9.11919e-05` |
