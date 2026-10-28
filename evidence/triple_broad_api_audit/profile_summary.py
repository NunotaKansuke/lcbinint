import json,re,collections
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'profiles';out=[]
def stage(s):
 s='\n'.join(line.replace('(anonymous namespace)','anonymous_namespace').split('(',1)[0]for line in s.splitlines())
 if re.search(r'regular_support_detail::(?:Tree|cover|kappa|critical)|caustic_clearance',s):return 'caustic_geometry'
 if re.search(r'PreparedRegularSupport::|regular_support_detail::|event_detail::|radial_event_plan',s):return 'image_support_events'
 if re.search(r'PointSourceMagnifier::|triple_approximation|triple_curvature_indicator|triple_image_candidates|triple_point',s):return 'point_hex_roots'
 if re.search(r'FirstOrderPanel|Patterson|Fejer|angular_detail|angular_topology|coefficient_detail|radius_terms|integrate|radial_panel|fold_leading|shape_gradient_detail|RadiusWorkspace|RadialPanel|continued_topology|resultant_detail|physical_detail|interval_detail',s):return 'integration'
 if 'triple_algebraic_epoch'in s:return 'router_inlined_other'
 if re.search(r'LensModel::|triple_algebraic_gradient|lcbinint::',s):return 'native_other'
 return 'unresolved_or_python_libraries'
def leaf_category(s):
 if re.search(r'malloc|free|operator new|operator delete|realloc|memcpy|memmove|memset',s):return 'allocation_memory'
 if re.search(r'quadmath|float128|cpp_bin_float|[a-z]+q\+|__.*tf3',s):return 'quad_math'
 if re.search(r'atan|hypot|sincos|sin\+|cos\+|sqrt|divdc3|complex',s):return 'scalar_complex_math'
 if re.search(r'\b_?Py|python3',s):return 'python'
 return 'other'
for m in json.loads((P/'profiles.json').read_text()):
 if m['returncode']!=0:continue
 stacks=(D/(m['key']+'.symbolized.stacks')).read_text().strip().split('\n\n');st=collections.Counter();lc=collections.Counter();leaves=collections.Counter();total=0
 for block in stacks:
  frames=[line.strip()for line in block.splitlines()[1:]if line.strip()]
  if not frames:continue
  total+=1;st[stage('\n'.join(frames))]+=1;lc[leaf_category(frames[0])]+=1;leaf=re.sub(r'\+0x[0-9a-f]+(?= \()', '',re.sub(r'^[0-9a-f]+ ', '',frames[0]));leaves[leaf]+=1
 z=dict(key=m['key'],condition=m['condition'],samples=total,stage_samples=dict(st),stage_percent={k:100*v/total for k,v in st.items()},leaf_percent={k:100*v/total for k,v in lc.items()},top_leaf_symbols=[dict(symbol=k,samples=v,percent=100*v/total)for k,v in leaves.most_common(12)])
 out.append(z)
(P/'profile-summary.json').write_text(json.dumps(out,indent=2)+'\n')
for z in out:print(z['key'],z['samples'],{k:round(v,1)for k,v in z['stage_percent'].items()})
