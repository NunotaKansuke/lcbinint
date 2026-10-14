#define main phase78_reference_main
#include "../moving_map_gradient_phase72/rand035_reference/reference.cpp"
#undef main

namespace {
double direct_ld_density(double R, const LensParams& p, const Rule& rule,
                         bool& ok) {
    const PrimaryFrame pf = PrimaryFrame::from(p);
    RefValue diag;
    const RefArcs arcs = certified_arcs(R, pf, diag);
    if (diag.fail || arcs.kind == ArcKind::kDegenerate) {
        ok = false;
        return NAN;
    }
    if (arcs.kind == ArcKind::kEmpty) return 0.0;
    Q flux = Q(0);
    const Q pi = acosq(Q(-1));
    if (arcs.kind == ArcKind::kFull) {
        for (int j = 0; j < static_cast<int>(rule.x.size()); ++j) {
            const Q theta = pi * (Q(rule.x[j]) + Q(1));
            const Q ph = qphi(Q(R), theta, pf);
            if (!(ph > Q(0))) { ok = false; return NAN; }
            flux += Q(rule.w[j]) * sqrtq(ph) * pi;
        }
    } else {
        for (const auto& arc : arcs.arcs) {
            const Q half = (arc.hi - arc.lo) / Q(2);
            const Q mid = (arc.hi + arc.lo) / Q(2);
            for (int j = 0; j < static_cast<int>(rule.x.size()); ++j) {
                const Q z = pi * Q(rule.x[j]) / Q(2);
                const Q theta = mid + half * sinq(z);
                const Q jac = half * pi * cosq(z) / Q(2);
                const Q ph = qphi(Q(R), theta, pf);
                if (!(ph > Q(0))) { ok = false; return NAN; }
                flux += Q(rule.w[j]) * jac * sqrtq(ph);
            }
        }
    }
    const Q den = Q(2) * pi * Q(p.rho) * Q(p.rho) / Q(3);
    return static_cast<double>(Q(R) * flux / den);
}

double local_fd(double R, const LensParams& p, double h,
                const Rule& rule, bool& ok) {
    std::array<double,4> vals{};
    const std::array<int,4> offsets{{-2,-1,1,2}};
    for (int i=0;i<4;++i) {
        LensParams perturbed=p;
        perturbed.xs += offsets[i]*h;
        bool valid=true;
        vals[i]=direct_ld_density(R,perturbed,rule,valid);
        ok=ok&&valid&&std::isfinite(vals[i]);
    }
    return ok ? (vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*h) : NAN;
}
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    LensParams p;
    bool found=false;
    std::ifstream in(argv[1]);
    std::string line,name;
    int row=0,barycentric=0;
    double input_u=0,mu=0;
    while(std::getline(in,line)) {
        ++row;
        std::istringstream ss(line);
        if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>barycentric>>input_u>>mu>>name))continue;
        if(row==17&&name=="caustic-cross"){found=true;break;}
    }
    if(!found)return 3;
    p.barycentric=barycentric!=0;
    TopologyResult topo=classify_cells(PrimaryFrame::from(p),nullptr,nullptr,true);
    if(topo.status!=Status::OK)return 4;
    adaptive_detail::certify_projective_p4_events(topo,PrimaryFrame::from(p));
    std::vector<CellPlan> cells;
    if(!adaptive_detail::restore_physical_cuts(topo,PrimaryFrame::from(p),cells))return 5;
    auto physical=[&](double R) {
        for(const auto& e:topo.events)
            if(e.radius==R&&adaptive_detail::adaptive_physical_fold_event(e))return true;
        return false;
    };
    const Rule ref_rule(128);
    const double D=kPi*p.rho*p.rho*(1.0-1.0/3.0);
    AdaptiveConfig cfg;
    std::ofstream out(argv[2],std::ios::trunc);
    if(!out)return 6;
    out<<std::setprecision(17)
       <<"cell\tkind\tleft_fold\tright_fold\tx\tR\tanalytic_dx\t"
          "production_fd_h1e-3\tproduction_fd_h2.5e-4\t"
          "direct_fd_h1e-3\tdirect_fd_h2.5e-4\treliable\t"
          "inner_estimate\n";
    for(const auto& cell:cells) {
        if(cell.kind!=ArcKind::kArcs&&cell.kind!=ArcKind::kFull)continue;
        const bool left=physical(cell.r_lo),right=physical(cell.r_hi);
        const FoldRadialMap map{cell.r_lo,cell.r_hi,left,right};
        for(double x:{-0.6,0.0,0.6}) {
            const double R=map(x)[0];
            const auto analytic=adaptive_detail::mapped_radius(
                R,1.0,p,1.0,PrimaryFrame::from(p),cell,true,false,nullptr,
                &cfg,0.0,false,nullptr);
            auto production_value=[&](double shift,bool& ok) {
                LensParams pp=p;pp.xs+=shift;
                const auto s=adaptive_detail::mapped_radius(
                    R,1.0,pp,1.0,PrimaryFrame::from(pp),cell,false,false,nullptr,
                    &cfg,0.0,false,nullptr);
                ok=s.reliable&&std::isfinite(s.value[0]);
                return s.value[0];
            };
            auto prod5=[&](double h,bool& ok) {
                std::array<double,4> vals{};
                const std::array<int,4> off{{-2,-1,1,2}};
                for(int i=0;i<4;++i){bool good=false;vals[i]=production_value(off[i]*h,good);ok=ok&&good;}
                return ok?(vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*h):NAN;
            };
            bool prod_ok=true,ref_ok1=true,ref_ok2=true;
            const double h1=p.rho*1e-3,h2=p.rho*2.5e-4;
            const double prod1=prod5(h1,prod_ok),prod2=prod5(h2,prod_ok);
            const double ref1=local_fd(R,p,h1,ref_rule,ref_ok1);
            const double ref2=local_fd(R,p,h2,ref_rule,ref_ok2);
            out<<cell.index<<'\t'<<static_cast<int>(cell.kind)<<'\t'<<left<<'\t'
               <<right<<'\t'<<x<<'\t'<<R<<'\t'<<analytic.value[1]<<'\t'
               <<prod1<<'\t'<<prod2<<'\t'<<ref1<<'\t'<<ref2<<'\t'
               <<(analytic.reliable&&prod_ok&&ref_ok1&&ref_ok2)<<'\t'
               <<analytic.inner[1]<<'\n';
        }
    }
    (void)D;
    return 0;
}
