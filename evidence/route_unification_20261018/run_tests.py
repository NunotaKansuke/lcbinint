"""Run against an explicit local build despite installed editable finders."""
from pathlib import Path
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
root = Path(__file__).resolve().parents[2]
sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in getattr(f, "__module__", "")]
build = root / os.environ.get("LCBININT_AUDIT_BUILD", "build-route-unified")
sys.path.insert(0, str(build))
import lcbinint
import lcbinint_jax
assert Path(lcbinint.__file__).is_relative_to(build)
assert Path(lcbinint._native.__file__).is_relative_to(build)
assert Path(lcbinint_jax.__file__).is_relative_to(build)
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
