import sys
from pathlib import Path
sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable'not in str(f)]
sys.path.insert(0,str(Path(__file__).resolve().parent/'local-package'))
import lcbinint,pytest
print(lcbinint._native.__file__,flush=True)
raise SystemExit(pytest.main(['-q']+['tests/regression/'+f for f in ['test_triple_event_precision.py','test_triple_ld_contact_gradients.py','test_triple_local_images.py','test_triple_repaired_contact_packets.py','test_triple_contact_resolution.py','test_triple_faint_images.py','test_triple_expanded_contract.py','test_triple_endpoint_promotion.py','test_triple_default_pipeline.py','test_triple_cusp_component.py']]))
