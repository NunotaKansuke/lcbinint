[Previous: Orbital motion](OrbitalMotion.md) · [Documentation home](readme.md) · [Next: Combining higher-order effects](CombinedEffects.md)

# Binary sources

`source="binary"` describes two luminous source stars magnified by the same
lens.  It changes the source trajectory and flux model only; it is independent
of whether the configured lens is binary or triple.

The observed magnification is the flux-weighted sum

$$
A(t) = \frac{A_1(t) + f\,A_2(t)}{1+f},
\qquad f = \mathtt{flux\_ratio} = F_2/F_1.
$$

`flux_ratio` is a photometric quantity.  It must not be used as a proxy for a
source mass ratio: stars of the same mass need not have the same brightness,
and vice versa.

## Static binary sources

Without xallarap, each source follows its own rectilinear trajectory.  They
share the lens parameters, `tE`, `alpha`, parallax, lens orbital motion, and
limb-darkening coefficients, but have separate closest-approach parameters and
source sizes.

| Parameter | Meaning |
| --- | --- |
| `t0`, `u0`, `rho1` | Closest-approach epoch, impact parameter, and normalized radius of source 1. |
| `t0_2`, `u0_2`, `rho2` | The corresponding quantities for source 2. |
| `tE`, `alpha` | Shared Einstein timescale and trajectory angle. |
| `flux_ratio` | Source flux ratio, \(F_2/F_1\). |

Both `rho1` and `rho2` are required.  Set either to zero for a point source.

```python
import numpy as np
import lcbinint

curve = lcbinint.LightCurve(source="binary")
parameters = {
    "s": 0.9, "q": 0.1, "alpha": 1.0, "tE": 30.0,
    "t0": 7500.0, "u0": 0.10, "rho1": 0.004,
    "t0_2": 7501.2, "u0_2": -0.06, "rho2": 0.002,
    "flux_ratio": 0.4,
}
times = np.linspace(7470.0, 7530.0, 300)
magnification = curve(times, parameters)
```

## Source orbital motion (xallarap)

With xallarap, the two sources are the two members of a physical source binary.
Their sky-plane positions are related to the source-binary centre of mass
(CoM):

$$
M_1\mathbf r_1 + M_2\mathbf r_2 = 0.
$$

For `source_mass_ratio = M_2/M_1`, once the state of source 1 is known, source
2 is fixed by

$$
\mathbf r_2 = -\frac{\mathbf r_1}{\mathtt{source\_mass\_ratio}}.
$$

This is why xallarap for a single source needs no mass ratio: only the
trajectory of the observed source is required.  A binary-source calculation
must also locate source 2, so it requires `source_mass_ratio`.  This parameter
controls dynamics only and remains independent of `flux_ratio`.

All binary-source xallarap models require:

```python
"rho1": 0.004,
"rho2": 0.002,
"flux_ratio": 0.4,
"source_mass_ratio": 0.7,  # M2 / M1, not a flux ratio
```

### Orbital-elements modes

`xallarap="circular_elements"` and `xallarap="orbital_elements"` use the
existing xallarap element parameterizations.  Supply the source-1 orbit state
through `xi_1`, `xi_2` and the relevant orbital parameters.  The second
source's state is generated using the CoM relation above.

| Mode | Required orbital parameters |
| --- | --- |
| `"circular_elements"` | `xi_1`, `xi_2`, `period_xa`, `inc_xa` |
| `"orbital_elements"` | `xi_1`, `xi_2`, `period_xa`, `ecc_xa`, `peri_xa`, `inc_xa` |

There is no `source_orbit_coordinates` switch for elements modes, and do not
provide `t0_2` or `u0_2`: positions are specified by the orbit itself.

```python
curve = lcbinint.LightCurve(
    source="binary",
    xallarap="circular_elements",
    t_ref=7500.0,
)
parameters = {
    "s": 0.9, "q": 0.1, "alpha": 1.0, "tE": 30.0,
    "t0": 7500.0, "u0": 0.10,
    "rho1": 0.004, "rho2": 0.002, "flux_ratio": 0.4,
    "source_mass_ratio": 0.7,
    "xi_1": 0.04, "xi_2": -0.02,
    "period_xa": 120.0, "inc_xa": 0.8,
}
```

### Position--velocity modes

`xallarap="circular_velocity"` and `xallarap="kepler_velocity"` propagate
an orbit from a projected position and velocity at `t_ref`.  For a binary
source, select how that initial position is supplied with
`source_orbit_coordinates`.

| Value | Position inputs | Use case |
| --- | --- | --- |
| `"xallarap"` | `t0`, `u0` specify the CoM trajectory; `xi_1`, `xi_2` specify source 1 relative to the CoM. | Direct state-vector sampling. |
| `"trajectory_offset"` | `t0`, `u0`, `t0_2`, `u0_2` specify the two source trajectories at the reference epoch. | Fits that conventionally sample two source tracks. |

Both choices require `t_ref`, `source_mass_ratio`, and the velocity parameters
`w1`, `w2`, `w3`.  `kepler_velocity` additionally requires `xa_szs` and
`xa_ar`.  These are the source-orbit state parameters of the existing velocity
xallarap API: `w1`, `w2`, and `w3` specify the instantaneous projected and
line-of-sight velocity state, while `xa_szs` and `xa_ar` complete the Kepler
geometry.

#### `source_orbit_coordinates="xallarap"`

This is the direct state-vector convention.  `t0` and `u0` belong to the CoM;
`xi_1` and `xi_2` are source 1's projected position relative to it at `t_ref`.
The second state follows from the mass ratio.

```python
curve = lcbinint.LightCurve(
    source="binary",
    xallarap="circular_velocity",
    source_orbit_coordinates="xallarap",
    t_ref=7500.0,
)
parameters = {
    "s": 0.9, "q": 0.1, "alpha": 1.0, "tE": 30.0,
    "t0": 7500.0, "u0": 0.10,
    "rho1": 0.004, "rho2": 0.002, "flux_ratio": 0.4,
    "source_mass_ratio": 0.7,
    "xi_1": 0.04, "xi_2": -0.02,
    "w1": 0.01, "w2": 0.8, "w3": 0.2,
}
```

#### `source_orbit_coordinates="trajectory_offset"`

This convention retains the familiar two-track inputs.  At `t_ref`, the
physical positions of source 1 and source 2 are exactly the positions obtained
from their respective rectilinear trajectories:

$$
\tau_i(t_\mathrm{ref}) = \frac{t_\mathrm{ref}-t_{0,i}}{t_E},
\qquad \beta_i(t_\mathrm{ref})=u_{0,i}.
$$

Internally, `lcbinint` converts them to a CoM trajectory and an orbital
relative state.  For \(q_s=M_2/M_1\), the conversion is

$$
t_{0,\mathrm{CoM}}=\frac{t_{0,1}+q_s t_{0,2}}{1+q_s},\qquad
u_{0,\mathrm{CoM}}=\frac{u_{0,1}+q_s u_{0,2}}{1+q_s},
$$

with the source-1 relative position chosen so that both positions above are
preserved at `t_ref`.  Consequently, the supplied `t0`, `u0`, `t0_2`, and
`u0_2` are not silently reinterpreted at the reference epoch.  Once orbital
motion is present, they are tangent-track parameters rather than exact
closest-approach times of the curved trajectories away from `t_ref`.

```python
curve = lcbinint.LightCurve(
    source="binary",
    xallarap="circular_velocity",
    source_orbit_coordinates="trajectory_offset",
    t_ref=7500.0,
)
parameters = {
    "s": 0.9, "q": 0.1, "alpha": 1.0, "tE": 30.0,
    "t0": 7500.0, "u0": 0.10,
    "t0_2": 7501.2, "u0_2": -0.06,
    "rho1": 0.004, "rho2": 0.002, "flux_ratio": 0.4,
    "source_mass_ratio": 0.7,
    "w1": 0.01, "w2": 0.8, "w3": 0.2,
}
```

Choose `t_ref` near the event centre or the centre of the data span.  This
makes the stated positional anchor useful and usually reduces parameter
correlations.

## Removed legacy names

The legacy names `q_mass`, `q_source`, and `fluxratio` are rejected.  Use
`source_mass_ratio` for \(M_2/M_1\) only when binary-source xallarap is
active, and `flux_ratio` for \(F_2/F_1\) in every binary-source model.

[Previous: Orbital motion](OrbitalMotion.md) · [Documentation home](readme.md) · [Next: Combining higher-order effects](CombinedEffects.md)
