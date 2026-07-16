import math

import pytest


@pytest.mark.parametrize(
    ("relation", "kwargs", "intercept", "slope", "magnitude", "color"),
    [
        ("VI", {"I": 18.0, "V_I": 1.2}, 0.5014135, 0.41968496, 18.0, 1.2),
        ("IH", {"I": 18.0, "I_H": 1.4}, 0.53026, 0.36595, 18.0, 1.4),
        ("JH", {"H": 16.0, "J_H": 0.6}, 0.5013, 0.4312, 16.0, 0.6),
        ("VH", {"H": 16.0, "V_H": 2.6}, 0.5145, 0.0892, 16.0, 2.6),
    ],
)
def test_theta_star_relation_median(relation, kwargs, intercept, slope, magnitude, color):
    lcbinint = pytest.importorskip("lcbinint")

    log_median, log_sigma = getattr(lcbinint.theta_star, relation)(**kwargs)
    expected = 0.5 * 10.0 ** (intercept + slope * color - 0.2 * magnitude)

    assert math.exp(log_median) == pytest.approx(expected)
    assert log_sigma > 0.0


def test_theta_star_relation_propagates_measurement_errors():
    lcbinint = pytest.importorskip("lcbinint")

    _, base_sigma = lcbinint.theta_star.IH(I=18.0, I_H=1.4)
    _, measured_sigma = lcbinint.theta_star.IH(
        I=18.0,
        I_H=1.4,
        I_error=0.1,
        I_H_error=0.2,
    )

    assert measured_sigma > base_sigma


@pytest.mark.parametrize(
    "kwargs",
    [
        {"I": float("nan"), "I_H": 1.4},
        {"I": 18.0, "I_H": 1.4, "I_error": -0.1},
    ],
)
def test_theta_star_relation_rejects_invalid_inputs(kwargs):
    lcbinint = pytest.importorskip("lcbinint")

    with pytest.raises(ValueError):
        lcbinint.theta_star.IH(**kwargs)
