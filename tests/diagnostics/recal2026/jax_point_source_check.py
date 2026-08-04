#!/usr/bin/env python3
"""Does the JAX backend's point-source magnification match the native one?

The JAX FFI solves the binary lens polynomial and classifies its roots in
``python/bind_jax_ir.cpp``, independently of the library's own solver.  Two
implementations of the same quantity are worth checking against each other on
their own, but the reason this exists is that the JAX one was silently wrong:
polynomial ghost roots that Newton-polish onto a physical image were being
counted as separate images, on 5.7% of sampled positions and by up to 94%,
with ``root_failure`` clear and ``topology_stable`` true throughout.

Two regimes are sampled because the failure and the thing that must not break
live in different places.  The wide field is where the ghosts land on top of
physical images and where every observed failure was.  The neighbourhood of the
central caustic is where genuine five-image solutions live, and a deduplication
rule that is too eager would destroy them without the wide field noticing.
"""

from __future__ import annotations

import argparse

import numpy as np


def _positions(rng, count, near_caustic):
    """Lens configurations and source positions, in the vbm frame."""
    for _ in range(count):
        s = float(10 ** rng.uniform(-0.5, 0.5))
        q = float(10 ** rng.uniform(-5, 0))
        if near_caustic:
            radius = 10 ** rng.uniform(-4, -0.5)
            angle = rng.uniform(0.0, 2.0 * np.pi)
            yield s, q, radius * np.cos(angle), radius * np.sin(angle)
        else:
            yield s, q, rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)


def run(count=1500, tolerance=1.0e-6, quiet=False):
    """Returns the worst relative gap and the failure count, per regime."""
    from .engines_ext import configure_jax

    configure_jax()
    import jax.numpy as jnp
    import lcbinint
    from lcbinint_jax.cpp_backend import binary_hexadecapole_batch_ffi

    native = lcbinint.LightCurve(lens="binary", options=lcbinint.Options(
        coordinates="vbm", nbin="auto", caustic_bins=1400, reltol=1.0e-3))
    # Small enough that the finite-source correction cannot hide a disagreement
    # in the point term, which is what this compares.
    source_radius = 1.0e-4
    summary = {}
    for label, near, seed in (("wide field", False, 4041),
                              ("central caustic", True, 777)):
        rng = np.random.default_rng(seed)
        worst = 0.0
        failures = 0
        worst_case = None
        for s, q, x, y in _positions(rng, count, near):
            # The FFI takes the lens-frame mass ratio, which vbm inverts.
            expansion = binary_hexadecapole_batch_ffi(
                jnp.asarray([x]), jnp.asarray([y]), jnp.asarray(s),
                jnp.asarray(1.0 / q), jnp.asarray(source_radius),
                jnp.asarray(0.0), jnp.asarray(0.0))
            traced = float(np.asarray(expansion.point_magnification)[0])
            if bool(np.asarray(expansion.root_failure)[0]):
                # A declared failure is not a silent one; the pipeline routes
                # around it, so it is not what this test is looking for.
                continue
            info = native.info(x, t0=0.0, tE=1.0, u0=y, alpha=0.0,
                               s=s, q=q, rho=source_radius,
                               limb_darkening_c=0.0)
            reference = float(
                np.asarray(info.point_source_magnifications).ravel()[0])
            gap = abs(traced - reference) / abs(reference)
            if gap > tolerance:
                failures += 1
            if gap > worst:
                worst = gap
                worst_case = (s, q, x, y, traced, reference)
        summary[label] = (failures, worst, worst_case)
        if not quiet:
            print(f"{label:16s} {failures:4d}/{count} above {tolerance:g}   "
                  f"worst relative gap {worst:.3e}")
            if worst_case and worst > tolerance:
                s, q, x, y, traced, reference = worst_case
                print(f"    worst: s={s:.4f} q={q:.3e} (x,y)=({x:+.5f},{y:+.5f})"
                      f"  jax={traced:.6f} native={reference:.6f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1500,
                        help="positions per regime")
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    arguments = parser.parse_args()
    summary = run(count=arguments.count, tolerance=arguments.tolerance)
    failures = sum(entry[0] for entry in summary.values())
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
