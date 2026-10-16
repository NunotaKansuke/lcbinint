# Updated Cartesian/Polar auto route and integrated replay

No Cartesian or Polar integral was rerun. The rule was refit from the saved matched per-method results, then those results were combined offline.

- Conditions: 14,432; in-domain: 14,320.
- Selected rule: `boost-d3-n40`, 273 split nodes.
- Event groups are split by the established SHA256(case_id) rule; test cases do not enter scripted fitting or validation scoring. Their summaries were inspected during exploratory development, so this is not a sealed blind confirmation.
- Candidate selection uses the smallest tree within 0.5% of best validation geometric penalized regret.
- Training labels use measured time and estimated acceptance; a non-accepted route costs 5x the slowest finite route time during fitting. Validation and test regret use a fixed 10x penalty.
- Reference error and VBM timing are reporting-only.

## Held-out route comparison

| rule | geometric penalized regret | Polar selected | test conditions |
|---|---:|---:|---:|
| previous rule | 1.1650 | 1612 | 3232 |
| updated model, train only | 1.1204 | 1156 | 3232 |
| updated model, train + validation | 1.1272 | 1152 | 3232 |

## Integrated results

The four six-panel figures and `integrated-joined.tsv` select one saved adaptive method for every condition. The speed ratio is `VBM / (selected cold method time + saved route-feature extraction time)`; above one favors the selected route. Cold method time includes the indexed seed-cache miss and adaptive integration. The decision-tree evaluation and shared point-root/caustic preparation are outside the ratio, so this is a kernel-level replay rather than a new public `LightCurve` timing.

| profile | target | old Polar | new Polar | old median R | new median R | old aggregate R | new aggregate R | old heldout R | new heldout R | new heldout pass | new heldout false accepts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 1e-03 | 1849/3608 | 1488/3608 | 0.303 | 0.3229 | 0.1992 | 0.2085 | 0.334 | 0.3508 | 833/836 | 2 |
| uniform | 1e-04 | 1934/3608 | 1507/3608 | 0.4486 | 0.4853 | 0.2777 | 0.2698 | 0.4732 | 0.5234 | 833/836 | 2 |
| linear | 1e-03 | 1950/3608 | 1488/3608 | 1.348 | 1.454 | 1.889 | 2.041 | 1.501 | 1.556 | 835/836 | 0 |
| linear | 1e-04 | 2031/3608 | 1507/3608 | 4.903 | 5.175 | 7.074 | 7.121 | 5.222 | 5.487 | 834/836 | 1 |

The all-data map includes development rows and is descriptive. Test cases are disjoint from fitting, but the test summaries were viewed during exploration and are not a sealed final check. Error agreement is empirical, not a formal certificate.

The generated model is saved at `/rogue1_8/nunota/lcbinint/tests/diagnostics/results/recal2026/adaptive-route-update-20261016/adaptive_route_model.hpp`. Active selector update: `True`.
