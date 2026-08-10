# Pure-kernel rho and d/rho speed relation

`R = t_VBM / t_LCB-in`; therefore `R > 1` means LCB-in is faster.
The plotted d/rho is the refined actual caustic distance, not merely
the requested sampling factor. Only external epsilon values 1e-3 and
1e-4 are included.

## Overall

| profile | epsilon | n | LCB-in wins | win rate | median R |
|---|---:|---:|---:|---:|---:|
| uniform | `0.001` | 1132 | 0 | 0.0% | 0.091 |
| uniform | `0.0001` | 512 | 0 | 0.0% | 0.038 |
| linear | `0.001` | 1122 | 693 | 61.8% | 1.351 |
| linear | `0.0001` | 646 | 431 | 66.7% | 1.576 |

## Rho bands

| profile | epsilon | rho band | n | LCB-in wins | win rate | median R |
|---|---:|---|---:|---:|---:|---:|
| uniform | `0.001` | $3\times10^{-5}\leq\rho<10^{-3}$ | 228 | 0 | 0.0% | 0.074 |
| uniform | `0.001` | $10^{-3}\leq\rho<10^{-2}$ | 288 | 0 | 0.0% | 0.106 |
| uniform | `0.001` | $10^{-2}\leq\rho<10^{-1}$ | 308 | 0 | 0.0% | 0.120 |
| uniform | `0.001` | $\rho\geq10^{-1}$ | 276 | 0 | 0.0% | 0.077 |
| uniform | `0.0001` | $3\times10^{-5}\leq\rho<10^{-3}$ | 160 | 0 | 0.0% | 0.022 |
| uniform | `0.0001` | $10^{-3}\leq\rho<10^{-2}$ | 184 | 0 | 0.0% | 0.035 |
| uniform | `0.0001` | $10^{-2}\leq\rho<10^{-1}$ | 140 | 0 | 0.0% | 0.057 |
| uniform | `0.0001` | $\rho\geq10^{-1}$ | 8 | 0 | 0.0% | 0.025 |
| linear | `0.001` | $3\times10^{-5}\leq\rho<10^{-3}$ | 220 | 101 | 45.9% | 0.924 |
| linear | `0.001` | $10^{-3}\leq\rho<10^{-2}$ | 288 | 180 | 62.5% | 1.291 |
| linear | `0.001` | $10^{-2}\leq\rho<10^{-1}$ | 308 | 238 | 77.3% | 1.889 |
| linear | `0.001` | $\rho\geq10^{-1}$ | 274 | 165 | 60.2% | 1.472 |
| linear | `0.0001` | $3\times10^{-5}\leq\rho<10^{-3}$ | 171 | 79 | 46.2% | 0.841 |
| linear | `0.0001` | $10^{-3}\leq\rho<10^{-2}$ | 200 | 131 | 65.5% | 1.569 |
| linear | `0.0001` | $10^{-2}\leq\rho<10^{-1}$ | 204 | 176 | 86.3% | 2.758 |
| linear | `0.0001` | $\rho\geq10^{-1}$ | 47 | 29 | 61.7% | 3.547 |

## Actual d/rho bands

| profile | epsilon | actual d/rho | n | LCB-in wins | win rate | median R |
|---|---:|---|---:|---:|---:|---:|
| uniform | `0.001` | 0–0.3 | 548 | 0 | 0.0% | 0.092 |
| uniform | `0.001` | 0.3–0.8 | 496 | 0 | 0.0% | 0.095 |
| uniform | `0.001` | 0.8–1.05 | 84 | 0 | 0.0% | 0.060 |
| uniform | `0.0001` | 0–0.3 | 300 | 0 | 0.0% | 0.039 |
| uniform | `0.0001` | 0.3–0.8 | 176 | 0 | 0.0% | 0.034 |
| uniform | `0.0001` | 0.8–1.05 | 36 | 0 | 0.0% | 0.020 |
| linear | `0.001` | 0–0.3 | 538 | 389 | 72.3% | 1.680 |
| linear | `0.001` | 0.3–0.8 | 496 | 281 | 56.7% | 1.187 |
| linear | `0.001` | 0.8–1.05 | 84 | 23 | 27.4% | 0.581 |
| linear | `0.0001` | 0–0.3 | 368 | 303 | 82.3% | 2.673 |
| linear | `0.0001` | 0.3–0.8 | 226 | 112 | 49.6% | 0.979 |
| linear | `0.0001` | 0.8–1.05 | 52 | 16 | 30.8% | 0.495 |

Figures:
`figures/R_vs_actual_d_over_rho_by_rho.png` and
`figures/R_vs_rho_colored_actual_d_over_rho.png`.
