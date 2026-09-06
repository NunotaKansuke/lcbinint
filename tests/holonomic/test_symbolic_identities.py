"""Pytest wrapper around checks/holonomic/symbolic_checks.py.

The heavy lifting (re-deriving every boxed formula of the design document
from the lens equation) lives in that standalone script so it can also be
run by hand.  Here we just execute it and assert every identity passed,
surfacing the individual results as sub-checks.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPT = os.path.join(_ROOT, "checks", "holonomic", "symbolic_checks.py")
_JSON = os.path.join(_ROOT, "evidence", "holonomic", "symbolic_checks.json")


@pytest.fixture(scope="module")
def results():
    proc = subprocess.run([sys.executable, "-u", _SCRIPT], cwd=_ROOT,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"symbolic_checks.py exited {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-2000:]}")
    with open(_JSON) as fh:
        return json.load(fh)


def test_all_symbolic_identities_pass(results):
    assert results["all_pass"] is True


@pytest.mark.parametrize("name", [
    "P_is_quartic_in_t",
    "phi_equals_P_over_rho2R2AB_numeric",
    "disc_P_is_R4_times_D14_of_R2",
    "res_P_Pt_equals_p4_times_disc",
    "disc_A_is_minus4",
    "disc_B_is_minus4_R2ma2_sq",
    "res_A_B_is_16a2R2",
    "res_P_A_formula",
    "res_P_B_formula",
    "LD_reduction_identity",
    "H_degree_le_6",
    "gauss_manin_polynomial_identity",
    "rootpair_E_exact",
    "rootpair_O_exact",
    "rootpair_jacobian_det_at_v0",
    "Rmax_bound_no_image_outside",
])
def test_individual_identity(results, name):
    res = results["results"]
    if name not in res:
        pytest.skip(f"{name} not present in this revision of the check script")
    assert res[name]["ok"] is True, res[name]["detail"]
