import sys
from pathlib import Path
sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable'not in str(f)]
sys.path.insert(0,str(Path(__file__).resolve().parent/'candidate-package'))
import lcbinint,pytest
print(lcbinint._native.__file__,flush=True)
raise SystemExit(pytest.main(['-q','tests/regression/test_triple_ld_contact_gradients.py']))
