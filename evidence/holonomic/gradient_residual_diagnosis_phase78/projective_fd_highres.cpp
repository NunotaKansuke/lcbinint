#define main phase78_reference_main
#include "../moving_map_gradient_phase72/rand035_reference/reference.cpp"
#undef main

int main(int argc,char**argv) {
    if(argc!=3)return 2;
    LensParams base;bool found=false;std::ifstream in(argv[1]);
    std::string line,name;int row=0,bary=0;double in_u=0,mu=0;
    while(std::getline(in,line)) {
        ++row;std::istringstream ss(line);
        if(!(ss>>base.xs>>base.ys>>base.rho>>base.q>>base.a>>bary>>in_u>>mu>>name))continue;
        if(row==17&&name=="caustic-cross"){found=true;break;}
    }
    if(!found)return 3;base.barycentric=bary!=0;
    const std::array<Resolution,2> resolutions{{{128,4,256},{160,8,512}}};
    const std::array<double,2> hscales{{1e-3,2.5e-4}};
    const std::array<int,4> offsets{{-2,-1,1,2}};
    std::ofstream out(argv[2],std::ios::trunc);if(!out)return 4;
    out<<std::setprecision(17)
       <<"type\tresolution\thscale\toffset\tmu0\tmuhalf\tfail\t"
          "fd_mu0\tfd_muhalf\tfd_pure_ld_u1\n";
    for(const auto& res:resolutions)for(double hs:hscales) {
        const double h=base.rho*hs;
        std::array<RefValue,4> vals{};
        bool good=true;
        for(int i=0;i<4;++i) {
            LensParams p=base;p.xs+=offsets[i]*h;
            auto raw=classify_cells(PrimaryFrame::from(p),nullptr,nullptr,true);
            TopologyResult topo=raw;
            adaptive_detail::certify_projective_p4_events(topo,PrimaryFrame::from(p));
            int promoted=0;
            for(auto& e:topo.events)if(e.projective_fold_certified) {
                ++promoted;e.physically_real=true;e.kind="physical_real";
            }
            vals[i]=integrate_direct(p,topo,res);
            good=good&&vals[i].fail==0&&std::isfinite(vals[i].mu0)&&
                 std::isfinite(vals[i].mu_half)&&promoted==2;
            out<<"VALUE\tGL"<<res.nr<<'x'<<res.subdivisions<<'x'<<res.angular
               <<'\t'<<hs<<'\t'<<offsets[i]<<'\t'<<vals[i].mu0<<'\t'
               <<vals[i].mu_half<<'\t'<<vals[i].fail<<"\tNaN\tNaN\tNaN\n";
            out.flush();
        }
        double fd0=NAN,fdh=NAN,fd1=NAN;
        if(good) {
            auto fd=[&](double a,double b,double c,double d) {
                return (a-8*b+8*c-d)/(12*h);
            };
            fd0=fd(vals[0].mu0,vals[1].mu0,vals[2].mu0,vals[3].mu0);
            fdh=fd(vals[0].mu_half,vals[1].mu_half,vals[2].mu_half,vals[3].mu_half);
            fd1=2.5*fdh-1.5*fd0;
        }
        out<<"FD\tGL"<<res.nr<<'x'<<res.subdivisions<<'x'<<res.angular
           <<'\t'<<hs<<"\t0\tNaN\tNaN\t0\t"<<fd0<<'\t'<<fdh<<'\t'
           <<fd1<<'\n';
        out.flush();
    }
}
