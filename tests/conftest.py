import sys
from pathlib import Path

# Prefer the local build over any system-installed lcbinint.
_BUILD = Path(__file__).parent.parent / "build"
if _BUILD.exists():
    sys.path.insert(0, str(_BUILD))
