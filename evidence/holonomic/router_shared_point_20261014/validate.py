"""Validate the selected build without the host's editable-install redirects."""

import importlib.util
import os
from pathlib import Path
import sys

os.environ.update(JAX_ENABLE_X64="1", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
root = Path(__file__).resolve().parents[3]
build = root / "build-seeded-events/python-api"
for finder in list(sys.meta_path):
    known = getattr(finder, "known_source_files", {})
    if any(name == "lcbinint" or name.startswith("lcbinint.") or
           name == "lcbinint_jax" or name.startswith("lcbinint_jax.") for name in known):
        sys.meta_path.remove(finder)
sys.path.insert(0, str(build))
package = build / "lcbinint"
spec = importlib.util.spec_from_file_location("lcbinint", package / "__init__.py",
                                           submodule_search_locations=[str(package)])
module = importlib.util.module_from_spec(spec)
sys.modules["lcbinint"] = module
native_spec = importlib.util.spec_from_file_location(
    "lcbinint._lcbinint", next(package.glob("_lcbinint*.so")))
native = importlib.util.module_from_spec(native_spec)
sys.modules["lcbinint._lcbinint"] = native
native_spec.loader.exec_module(native)
spec.loader.exec_module(module)
print("Validated native:", native.__file__, flush=True)
os.chdir(root)
import pytest

raise SystemExit(pytest.main([
    "-q", "tests/regression/test_binary_caustic_enclosure.py",
    "tests/regression/test_point_source_safety.py",
    "tests/regression/test_smooth_route_clearance.py::test_binary_smooth_routes_stay_clear_of_the_caustic",
    "tests/regression/test_light_curve_warmup.py",
    "tests/regression/test_binary_source_components.py",
    "tests/regression/test_binary_ray_shooting.py",
    # This benchmark build has LCBININT_ENABLE_JAX_FFI=OFF. Its optional JAX
    # integration test was attempted and failed because that backend is absent;
    # report it separately instead of claiming the whole suite passed.
    "-k", "not test_jax_warmup_aligns_auto_route_and_compiles_value",
]))
