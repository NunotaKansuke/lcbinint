from pathlib import Path
import shutil,hashlib,json
root=Path(__file__).resolve().parents[3];build=root/'build-algebraic-boundary/gradient-probe'
for source,name in [('api-include','allmotion-include'),('direct256-include','direct_allmotion-include')]:
 target=build/name;shutil.copytree(build/source,target,dirs_exist_ok=True)
 p=target/'fused_k.hpp';s=p.read_text()
 old='inline MaterialMap make_material_map(const Event& left,const Event& right,const PrimaryFrame& pf) {'
 assert s.count(old)==1;s=s.replace(old,'inline MaterialMap make_material_map(const Event& left,const Event& right,const PrimaryFrame& pf, bool left_fixed=false, bool right_fixed=false) {')
 old='map.valid=derivative(left,map.da)&&derivative(right,map.db);'
 assert s.count(old)==1;s=s.replace(old,'map.valid=(left_fixed || derivative(left,map.da))&&(right_fixed || derivative(right,map.db));')
 p.write_text(s)
 p=target/'lcbinint/magnification/holonomic/adaptive_epoch.hpp';s=p.read_text()
 old='        bool weighted_candidate=cfg.weighted_radial_spans==nullptr;'
 addition='''        if(policy!=GradientPolicy::None) {
            adaptive_detail::EventLocation left{},right{};
            left.radius=panel.map.a;left.radius_lo=panel.left_radius_lo;
            right.radius=panel.map.b;right.radius_lo=panel.right_radius_lo;
            if(physical(c.r_lo))left=uncertainty(c.r_lo);
            if(physical(c.r_hi))right=uncertainty(c.r_hi);
            material_maps[i]=fused_gradient::make_material_map(
                left,right,pf,!physical(c.r_lo),!physical(c.r_hi));
        }
'''
 assert s.count(old)==1;s=s.replace(old,addition+old)
 a=s.index('inline AdaptiveSample mapped_full_circle(');b=s.index('\ninline ',a+10)
 sec=s[a:b]
 old='        const auto dh=internal_to_user_jac(dfh,p);'
 addition='''        std::array<double,5> df0{};
        if(const auto* motion=fused_gradient::active_motion)for(int j=0;j<5;++j) {
            df0[j]=jac*motion->radius[j]*kTwoPi+motion->jacobian[j]*f0;
            dfh[j]+=jac*motion->radius[j]*(fh/R)+motion->jacobian[j]*fh;
        }
        const auto d0=internal_to_user_jac(df0,p);
'''
 assert sec.count(old)==1;sec=sec.replace(old,addition+old)
 old='out.value[j+1]=(u*dh[j])/D;';assert sec.count(old)==1;sec=sec.replace(old,'out.value[j+1]=((1-u)*d0[j]+u*dh[j])/D;')
 p.write_text(s[:a]+sec+s[b:])
 (build/(name+'_manifest.json')).write_text(json.dumps({'source':source,'header_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2)+'\n')
