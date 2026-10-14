# Forward gradient experiment, 2026-10-14

The user explicitly deferred derivative accuracy. This experiment carries
five forward tangents on the value-selected mesh, skips derivative-only
refinement and derivative error estimation, and leaves their precision
uncertified. It is an isolated optional build; original source headers and
the normal value entry point are untouched.

Final matched benchmark: all 7,216 saved paper inputs, three repetitions,
mu_rtol=1e-4 and mu_atol=1e-8. Fresh seed/index preparation is included;
caustic prebuild is excluded. Value and fused evaluations share seeds but
use separate workspaces and alternate order. Run commands and compiler
flags are documented in the experiment's build script and README.

| Preparation-inclusive timing | Median per-input ratio | Ratio of summed per-input medians |
|---|---:|---:|
| All inputs | 1.268 | 1.239 |
| Uniform | 1.201 | 1.154 |
| Linear LD | 1.370 | 1.314 |

The target of 1.3x is attained in the overall median and aggregate timing,
but not for every input: 3,074 inputs exceed it, p95 is 1.649x, and maximum
is 2.110x. Excluding shared seed preparation, the summed frontier/solver
time ratio is 1.362x. These are observed timings on the host and backend
identified in provenance.json, not universal performance guarantees.

The original vs overlaid **value-only** computation matches exactly for
every input: magnification, convergence, stopping status, node count and
support. The fused primal differs in 300 inputs (284 node-count changes),
with maximum |difference|/max(1,|mu|) of 2.969e-8. No difference exceeds the
unchanged value tolerance and all primal/support stopping statuses match.
The angular quadrature and fallback paths differ when tangents are active;
fused primal equality is therefore tolerance-level, not bitwise.

The last optimization removes an unused derivative-error coordinate map.
Compared with full_fused_before_error_map_skip.tsv, all primal quantities,
nodes and statuses match exactly; tangent differences are at rounding scale
(maximum scaled difference 4.144e-12). This is a regression check, not an
independent derivative accuracy certification.

`full_baseline.tsv`, `full_fused.tsv`, `audit.json`, `provenance.json`,
and the build log are final evidence. Earlier pilot tables, trace files and
finite-difference probes are retained as exploratory history. Warm-index
pilot timings (`baseline.tsv`, `coupled.tsv`, `material.tsv`,
`material_fast.tsv`) must not be substituted for the final timing boundary.
Finite-difference probes often have unconverged reference values and are
not treated as certified derivative references.

Reproduce the final audit from the experiment checkout:

```bash
python benchmarks/holonomic/fused_gradient/audit.py evidence/holonomic/fused_gradient_20261014
python benchmarks/holonomic/fused_gradient/provenance.py evidence/holonomic/fused_gradient_20261014
```
