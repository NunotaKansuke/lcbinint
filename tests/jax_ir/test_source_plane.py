import jax
import numpy as np

from lcbinint_jax import binary_source_plane_quadrature

jax.config.update("jax_enable_x64", True)


PARAMETERS = (0.3, 0.4, 1.4, 1.0e-3, 0.01, 0.3, 0.2)


def test_ring_and_chord_rules_agree_for_a_smooth_source():
    ring = binary_source_plane_quadrature(
        *PARAMETERS,
        rule="ring",
        coarse_order=8,
        fine_order=16,
    )
    chord = binary_source_plane_quadrature(
        *PARAMETERS,
        rule="chord",
        coarse_order=8,
        fine_order=16,
    )
    assert bool(ring.converged)
    assert bool(chord.converged)
    assert not bool(ring.root_failure)
    assert not bool(chord.root_failure)
    np.testing.assert_allclose(ring.magnification, chord.magnification, rtol=5.0e-8)


def test_chord_rule_gradient_matches_finite_difference():
    def magnification(source_x):
        return binary_source_plane_quadrature(
            source_x,
            *PARAMETERS[1:],
            rule="chord",
            coarse_order=8,
            fine_order=16,
        ).magnification

    step = 1.0e-6
    finite_difference = (
        magnification(PARAMETERS[0] + step) - magnification(PARAMETERS[0] - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        jax.grad(magnification)(PARAMETERS[0]),
        finite_difference,
        rtol=2.0e-6,
        atol=1.0e-8,
    )


def test_chord_rule_rejects_an_unresolved_caustic_crossing():
    result = binary_source_plane_quadrature(
        0.653,
        0.0,
        1.2,
        0.1,
        0.02,
        0.4,
        0.0,
        rule="chord",
        coarse_order=8,
        fine_order=16,
    )
    assert not bool(result.converged)
    assert float(result.estimated_error) > 1.0e-2
