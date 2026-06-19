import pytest


ROOT_CASES = [
    pytest.param(
        [complex(-1.0, 0.0), 0.0j, 0.0j, complex(1.0, 0.0)],
        id="cubic_roots_of_unity",
    ),
    pytest.param(
        [
            complex(-1.0, 0.25),
            complex(0.5, -0.75),
            complex(-1.0, 0.0),
            complex(0.25, 0.5),
            complex(-0.25, 0.0),
            complex(1.0, 0.0),
        ],
        id="general_degree_5",
    ),
    pytest.param(
        [
            complex(0.2, -0.1),
            complex(-0.4, 0.3),
            complex(0.1, 0.2),
            complex(-0.7, 0.0),
            complex(0.4, -0.6),
            complex(-0.2, 0.1),
            complex(0.3, 0.0),
            complex(1.0, 0.0),
        ],
        id="general_degree_7",
    ),
]


def _sort_roots(roots):
    return sorted(roots, key=lambda root: (round(root.real, 12), round(root.imag, 12)))


def _vbm_roots(coefficients):
    module = pytest.importorskip("VBMicrolensing")
    vbm = module.VBMicrolensing()
    packed_coefficients = [[value.real, value.imag] for value in coefficients]
    return [complex(real, imag) for real, imag in vbm.cmplx_roots_gen(packed_coefficients)]


def _lcbinint_roots(coefficients):
    lcbinint = pytest.importorskip("lcbinint")
    if not hasattr(lcbinint, "polynomial_roots"):
        pytest.skip("lcbinint.polynomial_roots is not available")
    return lcbinint.polynomial_roots(coefficients)


@pytest.mark.parametrize("coefficients", ROOT_CASES)
def test_polynomial_roots_match_vbm(coefficients):
    reference = _sort_roots(_vbm_roots(coefficients))
    actual = _sort_roots(_lcbinint_roots(coefficients))

    assert len(actual) == len(reference)
    for actual_root, reference_root in zip(actual, reference):
        assert actual_root == pytest.approx(reference_root, rel=1.0e-11, abs=1.0e-11)
