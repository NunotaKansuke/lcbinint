from pathlib import Path
p=Path('src/lcbinint/magnification/holonomic/radius_terms.hpp');s=p.read_text()
s=s.replace('const PrimaryFrame& pf) {\n    if (ps.size() != n_expect)','const PrimaryFrame& pf,bool reciprocal=false) {\n    if (ps.size() != n_expect)',1)
s=s.replace('const C z{R*((1.0-t*t)/den), R*((2.0*t)/den)};','const C z{R*((reciprocal?t*t-1.0:1.0-t*t)/den), R*((reciprocal?-2.0*t:2.0*t)/den)};',1)
s=s.replace('double thm = (2.0 * std::atan(rp.m));\n        if (thm < 0.0) thm += kTwoPi;','double thm = (reciprocal?kPi:0.0)+2.0*std::atan(rp.m);\n        if (thm < 0.0) thm += kTwoPi;',1)
s=s.replace('bool valid = false;                 // pairs[] holds a usable previous-node set','bool valid = false;                 // pairs[] holds a usable previous-node set\n    bool reciprocal = false;            // u=-1/t for arcs crossing theta=pi',1)
a=s.index('inline std::vector<double> real_root_thetas_transport(');b=s.index('\n}',a)+2
part=s[a:b]
part=part.replace('    if (w.valid && !w.pairs.empty()) {','''    const auto transport_pc=w.reciprocal?boundary_quartic_reciprocal(QuarticCoeffs{pc}).p:pc;
    if(w.reciprocal && transport_pc[4]==0.0)w.valid=false;
    auto theta_from_coordinate=[&](double t) {
        double theta=(w.reciprocal?kPi:0.0)+2.0*std::atan(t);
        if(theta<0)theta+=kTwoPi;
        return theta;
    };
    auto radial_coeff=[&](bool reciprocal) {
        const auto derivative=boundary_quartic_dR(R,pf);
        return reciprocal?boundary_quartic_reciprocal(derivative).p:derivative.p;
    };
    if (w.valid && !w.pairs.empty()) {''',1)
part=part.replace('rp, pc, 8,','rp, transport_pc, 8,').replace('local_bracket_detail::certify(pc,endpoints,2)','local_bracket_detail::certify(transport_pc,endpoints,2)').replace('transport_pairs_valid(next, w.pairs.size(), pc, R, pf)','transport_pairs_valid(next, w.pairs.size(), transport_pc, R, pf,w.reciprocal)')
part=part.replace('(2*std::atan(pair.t_minus()))','theta_from_coordinate(pair.t_minus())').replace('(2*std::atan(pair.t_plus()))','theta_from_coordinate(pair.t_plus())').replace('double th = (2.0 * std::atan(t));','double th = theta_from_coordinate(t);')
part=part.replace('w.pc_prev=pc;w.pcR_prev=boundary_quartic_dR(R,pf).p;','w.pc_prev=transport_pc;w.pcR_prev=radial_coeff(w.reciprocal);')
part=part.replace('local_bracket_detail::certify(pc,tracked,count)','local_bracket_detail::certify(transport_pc,tracked,count)')
part=part.replace('w.pc_prev = pc;\n                    w.pcR_prev = boundary_quartic_dR(R, pf).p;','w.pc_prev = transport_pc;\n                    w.pcR_prev = radial_coeff(w.reciprocal);',1)
old='''        if (capacity_ok && transport_pairs_valid(cand, cand.size(), pc, R, pf)) {
            w.pairs = cand;
            w.pc_prev = pc;
            w.pcR_prev = boundary_quartic_dR(R, pf).p;
            w.R_prev = R;
            w.disc_sign = disc_now;
            w.valid = true;
            w.cold_streak = 0;
            seeded = true;
        }'''
new='''        auto accept_seed=[&](const RootPairSet& pairs,const std::array<double,5>& chart,bool reciprocal) {
            if(!transport_pairs_valid(pairs,pairs.size(),chart,R,pf,reciprocal))return false;
            w.pairs=pairs;w.pc_prev=chart;w.pcR_prev=radial_coeff(reciprocal);
            w.R_prev=R;w.disc_sign=disc_now;w.valid=true;
            w.reciprocal=reciprocal;w.cold_streak=0;
            return true;
        };
        if(capacity_ok)seeded=accept_seed(cand,pc,false);
#ifndef HOLO_MV_DIRECT_CHART_ONLY
        if(!seeded) {
            std::array<double,4> ur{};bool finite=true;
            for(std::size_t i=0;i<tr.size();++i) {
                ur[i]=-1.0/tr[i];if(!std::isfinite(ur[i]))finite=false;
            }
            if(finite) {
                std::sort(ur.begin(),ur.begin()+tr.size());
                RootPairSet reciprocal_pairs;bool capacity=true;
                for(std::size_t i=0;i+1<tr.size();i+=2)
                    capacity=reciprocal_pairs.push_back(root_pair_from_endpoints(ur[i],ur[i+1])) && capacity;
                const auto chart=boundary_quartic_reciprocal(QuarticCoeffs{pc}).p;
                if(capacity && chart[4]!=0.0)seeded=accept_seed(reciprocal_pairs,chart,true);
            }
        }
#endif'''
assert old in part;part=part.replace(old,new,1)
s=s[:a]+part+s[b:];p.write_text(s)
