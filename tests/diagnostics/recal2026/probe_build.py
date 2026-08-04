"""Import lcbinint from the instrumented build instead of the installed one.

The campaign's timing sweeps run for hours against the installed extension, and
rebuilding it in place would swap the shared object out from under them.  The
probe study therefore builds into its own tree and loads that, which also keeps
a build carrying calibration instrumentation from becoming what the rest of the
project imports by accident.

The package is installed in editable mode, which works through a meta-path
finder rather than ``sys.path``; prepending a directory is not enough to shadow
it, so the finder is removed first.  Import this module before ``lcbinint``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BUILD = Path(
    os.environ.get("LCBININT_PROBE_BUILD", "/rogue1_8/nunota/lcbinint/build_probe"))


def activate():
    """Point ``import lcbinint`` at the instrumented build, and return it."""
    if "lcbinint" in sys.modules:
        raise RuntimeError("lcbinint was already imported; activate() first")
    if not (BUILD / "lcbinint" / "__init__.py").exists():
        raise RuntimeError(f"no instrumented build at {BUILD}")
    sys.meta_path = [
        finder for finder in sys.meta_path
        if "editable" not in getattr(type(finder), "__module__", "")
    ]
    sys.path.insert(0, str(BUILD))
    import lcbinint

    resolved = Path(lcbinint.__file__).resolve()
    if BUILD.resolve() not in resolved.parents:
        raise RuntimeError(f"lcbinint still resolved to {resolved}")
    return lcbinint
