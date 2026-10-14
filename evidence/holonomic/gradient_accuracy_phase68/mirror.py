from pathlib import Path
import shutil
root=Path('/tmp/p68_headers/lcbinint/magnification/holonomic')
shutil.copytree('src/lcbinint/magnification/holonomic',root,dirs_exist_ok=True)
p=root/'adaptive_radial.hpp'
s=p.read_text().replace('if(p.level<cfg.max_level){','if(p.level<(gradient_phase?research_gradient_max_level:cfg.max_level)){')
import os
if os.environ.get('P68_FROZEN_PRIORITY')=='1':
 s=s.replace('!value_pass || !value_snapshot.valid','!value_snapshot.valid').replace('if(value_pass && gradient_pass)break;','if(value_snapshot.valid && gradient_pass)break;')
if os.environ.get('P68_GRAD_HYBRID')=='1':
 s=s.replace('j==0&&cfg.radial_error_estimator','cfg.radial_error_estimator')
p.write_text(s)
