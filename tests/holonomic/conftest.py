import os
import sys

# The reference package lives in the source tree, not the installed wheel.
_REF = os.path.join(os.path.dirname(__file__), "..", "..", "python", "lcbinint")
_REF = os.path.abspath(_REF)
if _REF not in sys.path:
    sys.path.insert(0, _REF)
