import sys
from pathlib import Path

# Prefer the current local build over any system-installed lcbinint.
for _name in ("build_new", "build"):
    _BUILD = Path(__file__).parent.parent / _name
    if _BUILD.exists():
        sys.path.insert(0, str(_BUILD))
        break
