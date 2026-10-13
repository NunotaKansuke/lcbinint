# Controlled pure-kernel speed/diagnostic map

Profile: `uniform`
Target: `0.0001`
Measured epoch points: `3608`
Input parts: `1` (read serially in numeric order)
Minimum cell population: `8`

The top row is the cell median of `R=t_VBM/t_lcbinint`; `R>1` favours lcbinint.
The bottom row is the cell p95 of the stored cross-code relative difference against VBM. It is a disagreement diagnostic, not an independent accuracy certificate.

| statistic | value |
|---|---:|
| lcbinint win rate (`R>1`) | `0.608%` |
| median `R` | `0.278373` |
| p10 `R` | `0.0176454` |
| p90 `R` | `0.682962` |
| cross-code error records | `3456` |
| cross-code error p95 | `0.000105385` |
