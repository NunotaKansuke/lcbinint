import sys
from pathlib import Path

# Prefer the current local build over any system-installed lcbinint.
# Editable installs register a meta-path finder ahead of PathFinder, so merely
# prepending the build directory does not shadow them.
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if "lcbinint" not in getattr(type(finder), "__module__", "")
    or "editable" not in getattr(type(finder), "__module__", "")
]
for _name in ("build", "build_new"):
    _BUILD = Path(__file__).parent.parent / _name
    if _BUILD.exists():
        sys.path.insert(0, str(_BUILD))
        break
