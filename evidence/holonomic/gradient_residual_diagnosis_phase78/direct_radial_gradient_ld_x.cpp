#define main phase78_reference_main
#include "../moving_map_gradient_phase72/rand035_reference/reference.cpp"
#undef main

#include <chrono>

namespace {
Q qphi_dx(Q R, Q theta, const PrimaryFrame& pf) {
    const Q ct=cosq(theta),st=sinq(theta);
    const QC z{R*ct,R*st},zb{R*ct,-R*st};
    const Q a=Q(pf.a),m0=Q(pf.m0),m1=Q(1)-m0,rho=Q(pf.rho);
    const QC iz=qdiv({Q(1),Q(0)},zb);
    const QC iza=qdiv({Q(1),Q(0)},qsub(zb,{a,Q(0)}));
    const QC f=qsub(qsub(z,{m0*iz.re,m0*iz.im}),{m1*iza.re,m1*iza.im});
    const QC g=qsub(f,{Q(pf.X),Q(pf.Y)});
    return Q(2)*g.re/(rho*rho);
}

bool projective_physical(const TopologyResult& topo,double R) {
    for(const auto& e:topo.events)
        if(e.radius==R && adaptive_detail::adaptive_physical_fold_event(e))return true;
    return false;
}

double fixed_R_dmu_dx(double R,double u,const LensParams& p,
                      const Rule& angular,bool& ok) {
    const PrimaryFrame pf=PrimaryFrame::from(p);
    RefValue diag;
    const RefArcs arcs=certified_arcs(R,pf,diag);
    if(diag.fail||arcs.kind==ArcKind::kDegenerate){ok=false;return NAN;}
    if(arcs.kind==ArcKind::kEmpty)return 0.0;
    const Q pi=acosq(Q(-1));
    Q dFh=Q(0);
    if(arcs.kind==ArcKind::kFull) {
        for(int i=0;i<static_cast<int>(angular.x.size());++i) {
            const Q theta=pi*(Q(angular.x[i])+Q(1));
            const Q ph=qphi(Q(R),theta,pf);
            if(!(ph>Q(0))){ok=false;return NAN;}
            dFh+=Q(angular.w[i])*qphi_dx(Q(R),theta,pf)/(Q(2)*sqrtq(ph))*pi;
        }
    } else {
        for(const auto& arc:arcs.arcs) {
            const Q half=(arc.hi-arc.lo)/Q(2),mid=(arc.hi+arc.lo)/Q(2);
            for(int i=0;i<static_cast<int>(angular.x.size());++i) {
                const Q z=pi*Q(angular.x[i])/Q(2);
                const Q theta=mid+half*sinq(z);
                const Q jac=half*pi*cosq(z)/Q(2);
                const Q ph=qphi(Q(R),theta,pf);
                if(!(ph>Q(0))){ok=false;return NAN;}
                dFh+=Q(angular.w[i])*jac*qphi_dx(Q(R),theta,pf)/
                     (Q(2)*sqrtq(ph));
            }
        }
    }
    const Q D=Q(2)*pi*Q(p.rho)*Q(p.rho)/Q(3);
    (void)u; // caller selects u=1, so the density is F_half / D.
    return static_cast<double>(Q(R)*dFh/D);
}

std::array<double,2> integrate_gradient(const LensParams& p,const TopologyResult& raw,
                          int nr,int radial_subdivisions,int ntheta,
                          bool& ok,
                          std::vector<std::array<double,2>>* by_cell=nullptr,
                          std::vector<std::array<std::uint64_t,7>>* k_by_cell=nullptr) {
    TopologyResult topo=raw;
    const PrimaryFrame pf=PrimaryFrame::from(p);
    adaptive_detail::certify_projective_p4_events(topo,pf);
    for(auto& e:topo.events)if(e.projective_fold_certified) {
        e.physically_real=true;
        e.kind="physical_real";
    }
    std::vector<CellPlan> cells;
    if(!adaptive_detail::restore_physical_cuts(topo,pf,cells)){ok=false;return {{NAN,NAN}};}
    const Rule radial(nr),angular(ntheta);
    long double total=0,total_production_pointwise=0;
    if(by_cell)by_cell->assign(cells.size(),std::array<double,2>{{0,0}});
    if(k_by_cell)k_by_cell->assign(cells.size(),std::array<std::uint64_t,7>{});
    AdaptiveConfig cfg;
    for(std::size_t cell_i=0;cell_i<cells.size();++cell_i) {
        const auto& cell=cells[cell_i];
        if(cell.kind==ArcKind::kEmpty)continue;
        V2Profile k_profile;
        V2ProfileScope profile_scope(k_profile);
        const FoldRadialMap map{cell.r_lo,cell.r_hi,
            projective_physical(topo,cell.r_lo),projective_physical(topo,cell.r_hi)};
        long double cell_total=0,cell_production=0;
        for(int sub=0;sub<radial_subdivisions;++sub) {
            const double xl=-1.0+2.0*sub/radial_subdivisions;
            const double xr=-1.0+2.0*(sub+1)/radial_subdivisions;
            const double xc=0.5*(xl+xr),xh=0.5*(xr-xl);
            for(int i=0;i<nr;++i) {
                const double x=xc+xh*radial.x[i];
                const auto rr=map(x);
                bool local_ok=true;
                const double density=fixed_R_dmu_dx(rr[0],1.0,p,angular,local_ok);
                const auto production=adaptive_detail::mapped_radius(
                    rr[0],1.0,p,1.0,pf,cell,true,false,nullptr,&cfg,0.0,false,nullptr);
                if(!local_ok||!std::isfinite(density)||!production.reliable||
                   !std::isfinite(production.value[1])){ok=false;return {{NAN,NAN}};}
                const long double weight=static_cast<long double>(xh*radial.w[i]*rr[1]);
                total+=weight*density;
                total_production_pointwise+=weight*production.value[1];
                cell_total+=weight*density;
                cell_production+=weight*production.value[1];
            }
        }
        if(by_cell)(*by_cell)[cell_i]={{static_cast<double>(cell_total),
                                       static_cast<double>(cell_production)}};
        if(k_by_cell)(*k_by_cell)[cell_i]={{k_profile.k_attempts,k_profile.k_success,
            k_profile.k_reject,k_profile.k_reject_vfloor,
            k_profile.k_reject_disagreement,k_profile.k_reject_s2,
            k_profile.k_reject_tmax}};
    }
    return {{static_cast<double>(total),static_cast<double>(total_production_pointwise)}};
}

template<int N>
double angular_rescue_dmu_dx(double R,const LensParams& p,const PrimaryFrame& pf,
                             const CellPlan& cell,bool& ok) {
    QuarticWarm quartic;
    RootPairWarm roots;
    QuarticCoeffs pc{};
    const auto arcs=adaptive_detail::adaptive_arc_intervals(
        R,pf,&quartic,&roots,&pc,cell.n_crossings);
    if(arcs.kind!=ArcKind::kArcs ||
       static_cast<int>(2*arcs.arcs.size())!=cell.n_crossings) {
        ok=false;
        return NAN;
    }
    const Cheb1<N> rule;
    std::array<double,5> dfh{};
    for(const auto& arc:arcs.arcs) {
        const auto pe=polish_endpoint(R,arc[0],pf,6);
        const auto pl=polish_endpoint(R,arc[1],pf,6);
        if(!pe.reliable||!pl.reliable) { ok=false; return NAN; }
        const double te=pe.theta;
        double tl=pl.theta;
        if(tl<=te)tl+=kTwoPi;
        const double half=0.5*(tl-te),mid=te+half;
        std::array<double,5> dh{};
        for(int k=0;k<N;++k) {
            const auto g=phi_val_dP(R,mid+half*rule.x[k],pf);
            if(!(g.phi>0.0)||!std::isfinite(g.phi)) { ok=false; return NAN; }
            const double z=0.5/std::sqrt(g.phi);
            for(int j=0;j<5;++j)dh[j]+=rule.w[k]*g.dP[j]*z;
        }
        for(int j=0;j<5;++j)dfh[j]+=R*half*dh[j];
    }
    const auto user=internal_to_user_jac(dfh,p);
    const double D=2.0*kPi*p.rho*p.rho/3.0;
    return user[0]/D;
}

template<int N>
double integrate_cell6_angular(const LensParams& p,const CellPlan& cell,
                               const FoldRadialMap& map) {
    const Rule radial(192);
    long double total=0;
    for(int sub=0;sub<8;++sub) {
        const double xl=-1.0+2.0*sub/8.0;
        const double xr=-1.0+2.0*(sub+1)/8.0;
        const double xc=0.5*(xl+xr),xh=0.5*(xr-xl);
        for(int i=0;i<192;++i) {
            const double x=xc+xh*radial.x[i];
            const auto rr=map(x);
            bool ok=true;
            const double g=angular_rescue_dmu_dx<N>(
                rr[0],p,PrimaryFrame::from(p),cell,ok);
            if(!ok||!std::isfinite(g))return NAN;
            total+=static_cast<long double>(xh*radial.w[i]*rr[1])*g;
        }
    }
    return static_cast<double>(total);
}

template<int N>
double sine_rescue_dmu_dx(double R,const LensParams& p,const PrimaryFrame& pf,
                          const CellPlan& cell,bool& ok) {
    QuarticWarm quartic;
    RootPairWarm roots;
    QuarticCoeffs pc{};
    const auto arcs=adaptive_detail::adaptive_arc_intervals(
        R,pf,&quartic,&roots,&pc,cell.n_crossings);
    if(arcs.kind!=ArcKind::kArcs ||
       static_cast<int>(2*arcs.arcs.size())!=cell.n_crossings) {
        ok=false;
        return NAN;
    }
    const Rule rule(N);
    std::array<double,5> dfh{};
    for(const auto& arc:arcs.arcs) {
        const auto pe=polish_endpoint(R,arc[0],pf,6);
        const auto pl=polish_endpoint(R,arc[1],pf,6);
        if(!pe.reliable||!pl.reliable) { ok=false; return NAN; }
        const double te=pe.theta;
        double tl=pl.theta;
        if(tl<=te)tl+=kTwoPi;
        const double half=0.5*(tl-te),mid=te+half;
        std::array<double,5> integral{};
        for(int k=0;k<N;++k) {
            const double z=0.5*kPi*rule.x[k];
            const double theta=mid+half*std::sin(z);
            const double dtheta_dx=half*0.5*kPi*std::cos(z);
            const auto g=phi_val_dP(R,theta,pf);
            if(!(g.phi>0.0)||!std::isfinite(g.phi)) { ok=false; return NAN; }
            const double w=rule.w[k]*dtheta_dx*0.5/std::sqrt(g.phi);
            for(int j=0;j<5;++j)integral[j]+=w*g.dP[j];
        }
        for(int j=0;j<5;++j)dfh[j]+=R*integral[j];
    }
    const auto user=internal_to_user_jac(dfh,p);
    const double D=2.0*kPi*p.rho*p.rho/3.0;
    return user[0]/D;
}

template<int N>
double integrate_cell6_sine(const LensParams& p,const CellPlan& cell,
                            const FoldRadialMap& map) {
    const Rule radial(192);
    long double total=0;
    for(int sub=0;sub<8;++sub) {
        const double xl=-1.0+2.0*sub/8.0;
        const double xr=-1.0+2.0*(sub+1)/8.0;
        const double xc=0.5*(xl+xr),xh=0.5*(xr-xl);
        for(int i=0;i<192;++i) {
            const double x=xc+xh*radial.x[i];
            const auto rr=map(x);
            bool ok=true;
            const double g=sine_rescue_dmu_dx<N>(
                rr[0],p,PrimaryFrame::from(p),cell,ok);
            if(!ok||!std::isfinite(g))return NAN;
            total+=static_cast<long double>(xh*radial.w[i]*rr[1])*g;
        }
    }
    return static_cast<double>(total);
}

struct KOrderValue {
    double value=NAN, dx=NAN, min_s2=INFINITY, min_b=INFINITY, max_abs_t=0;
    bool ok=false;
};

struct KOrderRule {
    int n=0;
    std::vector<double> x,w;
    KOrderRule()=default;
    explicit KOrderRule(int count):n(count),x(count),w(count) {
        for(int k=1;k<=n;++k) {
            const double alpha=k*kHoloPi/(n+1);
            x[k-1]=std::cos(alpha);
            const double sn=std::sin(alpha);
            w[k-1]=(kHoloPi/(n+1))*sn*sn;
        }
    }
};

KOrderValue eval_k_order(const KOrderRule& rule,double m,double v,double dm,double dv,
                         double R,const PrimaryFrame& pf,
                         const QuarticCoeffs& pc,
                         const QuarticParamJac& dpc,bool reciprocal) {
    KOrderValue out;
    const int n=rule.n;
    if(n<1||!(v>0))return out;
    const double s=std::sqrt(v);
    const DeflatedQuad q=deflate_mv(pc.p,m,v);
    const double p4=q.p4,d1=q.d1,d0=q.d0;
    const double dp4=dpc.dp[0][4];
    const double dd1=dpc.dp[0][3]+2.0*(dm*p4+m*dp4);
    const double dd0=dpc.dp[0][2]+2.0*(dm*d1+m*dd1)-
        (2.0*m*dm-dv)*p4-(m*m-v)*dp4;
    const double Rma=R-pf.a,Rpa=R+pf.a;
    const double b0=reciprocal?Rpa:Rma;
    const double b2=reciprocal?Rma*Rma:Rpa*Rpa;
    double G=0,dG=0;
    for(int k=1;k<=n;++k) {
        const double xi=rule.x[k-1],wi=rule.w[k-1];
        const double t=m+s*xi;
        const double S2=-(d0+t*(d1+t*p4));
        const double A=1+t*t,B=b0*b0+b2*t*t;
        out.min_s2=std::min(out.min_s2,S2);
        out.min_b=std::min(out.min_b,B);
        out.max_abs_t=std::max(out.max_abs_t,std::fabs(t));
        if(!(S2>0)||!(B>0))return out;
        const double wv=std::sqrt(S2)/(A*std::sqrt(A)*std::sqrt(B));
        G+=wi*wv;
        const double dt=dm+xi*(0.5/s)*dv;
        const double dS2=-(dd0+t*(dd1+t*dp4))-(d1+2*t*p4)*dt;
        const double dA=2*t*dt,dB=2*b2*t*dt;
        dG+=wi*wv*(0.5*dS2/S2-1.5*dA/A-0.5*dB/B);
    }
    out.value=(2.0/pf.rho)*v*G;
    out.dx=(2.0/pf.rho)*(dv*G+v*dG);
    out.ok=std::isfinite(out.value)&&std::isfinite(out.dx);
    return out;
}

struct KOrderAllValue {
    double value=NAN;
    std::array<double,5> jac{};
    double min_s2=INFINITY,min_b=INFINITY,max_abs_t=0;
    bool ok=false;
};

KOrderAllValue eval_k_order_all(const KOrderRule& rule,double m,double v,
    const std::array<double,5>& dm,const std::array<double,5>& dv,double R,
    const PrimaryFrame& pf,const QuarticCoeffs& pc,
    const QuarticParamJac& dpc,bool reciprocal) {
    KOrderAllValue out;
    if(rule.n<1||!(v>0))return out;
    const double s=std::sqrt(v),inv2s=0.5/s;
    const DeflatedQuad q=deflate_mv(pc.p,m,v);
    const double p4=q.p4,d1=q.d1,d0=q.d0;
    std::array<double,5> dp4{},dd1{},dd0{};
    for(int j=0;j<5;++j) {
        dp4[j]=dpc.dp[j][4];
        dd1[j]=dpc.dp[j][3]+2.0*(dm[j]*p4+m*dp4[j]);
        dd0[j]=dpc.dp[j][2]+2.0*(dm[j]*d1+m*dd1[j])-
            (2.0*m*dm[j]-dv[j])*p4-(m*m-v)*dp4[j];
    }
    const double Rma=R-pf.a,Rpa=R+pf.a;
    const double b0=reciprocal?Rpa:Rma;
    const double b2=reciprocal?Rma*Rma:Rpa*Rpa;
    double G=0;
    std::array<double,5> dG{};
    for(int k=0;k<rule.n;++k) {
        const double xi=rule.x[k],wi=rule.w[k],t=m+s*xi;
        const double S2=-(d0+t*(d1+t*p4));
        const double A=1+t*t,B=b0*b0+b2*t*t;
        out.min_s2=std::min(out.min_s2,S2);
        out.min_b=std::min(out.min_b,B);
        out.max_abs_t=std::max(out.max_abs_t,std::fabs(t));
        if(!(S2>0)||!(B>0))return out;
        const double wv=std::sqrt(S2)/(A*std::sqrt(A)*std::sqrt(B));
        G+=wi*wv;
        const double dS2_dt=-(d1+2.0*t*p4);
        const double dA_dt=2.0*t,dB_dt=2.0*b2*t;
        for(int j=0;j<5;++j) {
            const double dt=dm[j]+xi*inv2s*dv[j];
            const double dS2=-(dd0[j]+t*(dd1[j]+t*dp4[j]))+dS2_dt*dt;
            const double dA=dA_dt*dt;
            double dB=dB_dt*dt;
            if(j==4) dB+=reciprocal ? 2.0*Rpa-2.0*Rma*t*t
                                    : -2.0*Rma+2.0*Rpa*t*t;
            dG[j]+=wi*wv*(0.5*dS2/S2-1.5*dA/A-0.5*dB/B);
        }
    }
    const double two_over_rho=2.0/pf.rho;
    out.value=two_over_rho*v*G;
    for(int j=0;j<5;++j) {
        out.jac[j]=two_over_rho*(dv[j]*G+v*dG[j]);
        if(j==2)out.jac[j]+=-2.0/(pf.rho*pf.rho)*v*G;
    }
    out.ok=std::isfinite(out.value);
    for(double x:out.jac)out.ok=out.ok&&std::isfinite(x);
    return out;
}

double integrate_cell6_krule(int n,const LensParams& p,const CellPlan& cell,
                             const FoldRadialMap& map) {
    const Rule radial(192);
    const KOrderRule krule(n);
    const PrimaryFrame pf=PrimaryFrame::from(p);
    const double D=2.0*kPi*p.rho*p.rho/3.0;
    long double total=0;
    for(int sub=0;sub<8;++sub) {
        const double xl=-1.0+2.0*sub/8.0,xr=-1.0+2.0*(sub+1)/8.0;
        const double xc=0.5*(xl+xr),xh=0.5*(xr-xl);
        for(int i=0;i<192;++i) {
            const double x=xc+xh*radial.x[i];
            const auto rr=map(x);
            QuarticWarm qw;RootPairWarm rw;QuarticCoeffs arc_pc{};
            const auto arcs=adaptive_detail::adaptive_arc_intervals(
                rr[0],pf,&qw,&rw,&arc_pc,cell.n_crossings);
            if(arcs.kind!=ArcKind::kArcs)return NAN;
            const auto dpc0=boundary_quartic_dp(rr[0],pf);
            std::array<double,5> raw_jac{};
            for(const auto& arc:arcs.arcs) {
                auto te=polish_endpoint(rr[0],arc[0],pf,6);
                auto tl=polish_endpoint(rr[0],arc[1],pf,6);
                if(!te.reliable||!tl.reliable)return NAN;
                if(tl.theta<=te.theta)tl.theta+=kTwoPi;
                const std::array<double,5> zero{};
                bool accepted=false;
                for(bool reciprocal:{false,true}) {
                    const auto ap=reciprocal
                        ?arc_pair_jac_reciprocal(te.theta,tl.theta,zero,zero)
                        :arc_pair_jac(te.theta,tl.theta,zero,zero);
                    if(!ap.ok)continue;
                    const QuarticCoeffs cp=reciprocal
                        ?boundary_quartic_reciprocal(arc_pc):arc_pc;
                    const QuarticParamJac dp=reciprocal
                        ?boundary_quartic_reciprocal_dp(dpc0):dpc0;
                    std::array<double,5> dm{},dv{};
                    bool regular=true;
                    for(int j=0;j<5;++j) {
                        const auto d=root_pair_dR(RootPair{ap.m,ap.v},cp.p,dp.dp[j]);
                        regular=regular&&d.ok;
                        dm[j]=d.dm_dR;dv[j]=d.dv_dR;
                    }
                    if(!regular)continue;
                    const auto k=eval_k_order_all(krule,ap.m,ap.v,dm,dv,
                        rr[0],pf,cp,dp,reciprocal);
                    if(!k.ok)continue;
                    for(int j=0;j<5;++j)raw_jac[j]+=k.jac[j];
                    accepted=true;
                    break;
                }
                if(!accepted)return NAN;
            }
            for(double& z:raw_jac)z*=rr[1];
            const auto user_jac=internal_to_user_jac(raw_jac,p);
            total+=static_cast<long double>(xh*radial.w[i])*user_jac[0]/D;
        }
    }
    return static_cast<double>(total);
}

double qf_fixed_R_Fh(double R,const PrimaryFrame& pf,int ntheta,bool& ok) {
    RefValue diag;
    const RefArcs arcs=certified_arcs(R,pf,diag);
    if(diag.fail||arcs.kind!=ArcKind::kArcs){ok=false;return NAN;}
    const Rule rule(ntheta);
    const Q pi=acosq(Q(-1));
    Q total=0;
    for(const auto& arc:arcs.arcs) {
        const Q half=(arc.hi-arc.lo)/Q(2),mid=(arc.hi+arc.lo)/Q(2);
        for(int k=0;k<ntheta;++k) {
            const Q z=pi*Q(rule.x[k])/Q(2);
            const Q theta=mid+half*sinq(z);
            const Q ph=qphi(Q(R),theta,pf);
            if(!(ph>0)){ok=false;return NAN;}
            const Q jac=half*pi*cosq(z)/Q(2);
            total+=Q(R)*Q(rule.w[k])*jac*sqrtq(ph);
        }
    }
    return static_cast<double>(total);
}

struct QFArcResult { double value=NAN, dx=NAN; bool ok=false; };
QFArcResult qf_arc_Fh_dx(double R,Q lo,Q hi,const PrimaryFrame& pf,
                          double D,int ntheta) {
    const Rule rule(ntheta);
    const Q pi=acosq(Q(-1));
    const Q half=(hi-lo)/Q(2),mid=(hi+lo)/Q(2);
    Q value=0,dx=0;
    for(int k=0;k<ntheta;++k) {
        const Q z=pi*Q(rule.x[k])/Q(2);
        const Q theta=mid+half*sinq(z);
        const Q jac=half*pi*cosq(z)/Q(2);
        const Q ph=qphi(Q(R),theta,pf);
        if(!(ph>0))return {};
        value+=Q(rule.w[k])*jac*sqrtq(ph);
        dx+=Q(rule.w[k])*jac*qphi_dx(Q(R),theta,pf)/(Q(2)*sqrtq(ph));
    }
    return {static_cast<double>(Q(R)*value),
            static_cast<double>(Q(R)*dx/Q(D)),true};
}
}

int main(int argc,char**argv) {
    if(argc!=3)return 2;
    LensParams p;bool found=false;std::ifstream in(argv[1]);
    std::string line,name;int row=0,bary=0;double in_u=0,mu=0;
    while(std::getline(in,line)) {
        ++row;std::istringstream ss(line);
        if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>bary>>in_u>>mu>>name))continue;
        if(row==17&&name=="caustic-cross"){found=true;break;}
    }
    if(!found)return 3;p.barycentric=bary!=0;
    const auto topo=classify_cells(PrimaryFrame::from(p),nullptr,nullptr,true);
    if(topo.status!=Status::OK)return 4;
    std::ofstream out(argv[2],std::ios::trunc);if(!out)return 5;
    out<<std::setprecision(17)<<"nr\tradial_subdivisions\tntheta\tgrad_u1_dx_qf"
          "\tgrad_u1_dx_mapped_pointwise\tdifference\tok\n";
    double mapped_whole_baseline=NAN,mapped_cell6_baseline=NAN;
    for(const auto [nr,sub,ntheta]:{
        std::array<int,3>{{32,1,64}},std::array<int,3>{{64,2,64}},
        std::array<int,3>{{128,4,128}},std::array<int,3>{{192,8,128}}}) {
        bool ok=true;
        std::vector<std::array<double,2>> cell_values;
        std::vector<std::array<std::uint64_t,7>> cell_k_counts;
        const auto values=integrate_gradient(p,topo,nr,sub,ntheta,ok,
            (nr==192?&cell_values:nullptr),(nr==192?&cell_k_counts:nullptr));
        out<<nr<<'\t'<<sub<<'\t'<<ntheta<<'\t'<<values[0]<<'\t'
           <<values[1]<<'\t'<<values[1]-values[0]<<'\t'<<ok<<'\n';
        if(nr==192) {
            mapped_whole_baseline=values[1];
            mapped_cell6_baseline=cell_values.at(6)[1];
            TopologyResult marked=topo;
            adaptive_detail::certify_projective_p4_events(marked,PrimaryFrame::from(p));
            for(auto& e:marked.events)if(e.projective_fold_certified){e.physically_real=true;e.kind="physical_real";}
            std::vector<CellPlan> cells;
            adaptive_detail::restore_physical_cuts(marked,PrimaryFrame::from(p),cells);
            std::ofstream cells_out(std::string(argv[2])+".cells.tsv",std::ios::trunc);
            cells_out<<std::setprecision(17)<<"cell\tr_lo\tr_hi\tkind\tleft_fold\tright_fold"
                       "\tqf_direct\tmapped_pointwise\tdifference\tk_attempts"
                       "\tk_success\tk_reject\tk_vfloor\tk_disagreement\tk_s2\tk_tmax\n";
            for(std::size_t i=0;i<cells.size();++i) {
                const auto& c=cells[i];
                cells_out<<c.index<<'\t'<<c.r_lo<<'\t'<<c.r_hi<<'\t'
                    <<static_cast<int>(c.kind)<<'\t'
                    <<projective_physical(marked,c.r_lo)<<'\t'
                    <<projective_physical(marked,c.r_hi)<<'\t'
                    <<cell_values[i][0]<<'\t'<<cell_values[i][1]<<'\t'
                    <<cell_values[i][1]-cell_values[i][0];
                for(auto count:cell_k_counts[i])cells_out<<'\t'<<count;
                cells_out<<'\n';
            }
            std::ofstream events_out(std::string(argv[2])+".events.tsv",std::ios::trunc);
            events_out<<std::setprecision(17)<<"event\tradius\tkind\tphysical\tprojective"
                         "\tprecision_tier\tuncertainty\n";
            for(std::size_t i=0;i<marked.events.size();++i) {
                const auto& e=marked.events[i];
                events_out<<i<<'\t'<<e.radius<<'\t'<<e.kind<<'\t'
                    <<e.physically_real<<'\t'<<e.projective_fold_certified<<'\t'
                    <<e.precision_tier<<'\t'<<e.radius_uncertainty<<'\n';
            }
        }
        out.flush();
    }
    {
        TopologyResult marked=topo;
        adaptive_detail::certify_projective_p4_events(marked,PrimaryFrame::from(p));
        for(auto& e:marked.events)if(e.projective_fold_certified){e.physically_real=true;e.kind="physical_real";}
        std::vector<CellPlan> cells;
        adaptive_detail::restore_physical_cuts(marked,PrimaryFrame::from(p),cells);
        std::ofstream local(std::string(argv[2])+".cell6_local.tsv",std::ios::trunc);
        local<<std::setprecision(17)<<"x\tR\tjac\tqf_direct_dx\tmapped_dx\tdifference"
                "\tjac_weighted_difference\treliable\tk_attempts\tk_success"
                "\tk_reject_disagreement\tk_reject_vfloor\n";
        const Rule angular(256);
        const auto& cell=cells.at(6);
        const FoldRadialMap map{cell.r_lo,cell.r_hi,
            projective_physical(marked,cell.r_lo),projective_physical(marked,cell.r_hi)};
        AdaptiveConfig cfg;
        for(double x:{-0.5,0.0,0.5,0.75,0.9,0.95,0.98,0.99,0.995,0.999,0.9999}) {
            const auto rr=map(x);
            bool ref_ok=true;
            const double qf=fixed_R_dmu_dx(rr[0],1.0,p,angular,ref_ok);
            V2Profile profile;
            AdaptiveSample mapped;
            {
                V2ProfileScope scope(profile);
                mapped=adaptive_detail::mapped_radius(rr[0],1.0,p,1.0,
                    PrimaryFrame::from(p),cell,true,false,nullptr,&cfg,0.0,false,nullptr);
            }
            const double diff=mapped.value[1]-qf;
            local<<x<<'\t'<<rr[0]<<'\t'<<rr[1]<<'\t'<<qf<<'\t'
                 <<mapped.value[1]<<'\t'<<diff<<'\t'<<rr[1]*diff<<'\t'
                 <<(ref_ok&&mapped.reliable)<<'\t'<<profile.k_attempts<<'\t'
                 <<profile.k_success<<'\t'<<profile.k_reject_disagreement<<'\t'
                 <<profile.k_reject_vfloor<<'\n';
        }

        // Equation-derived GC2 K-rule convergence audit at the exact same
        // radial nodes. This is diagnostic only: it does not alter the
        // production 8/16 gate or select a new evaluator.
        std::ofstream korder(std::string(argv[2])+".cell6_krule.tsv",
                             std::ios::trunc);
        korder<<std::setprecision(17)
            <<"x\tR\tradial_jac\tarc\tchart\tm\tv\tmax_abs_t\tmin_s2\tmin_b"
              "\tK8\tK16\tK24\tK32\tK48\tK64\tK96\tK128\tK192\tK256\tK384\tK512\tK768\tK1024\tK1536\tK2048"
              "\tdX8\tdX16\tdX24\tdX32\tdX48\tdX64\tdX96\tdX128\tdX192\tdX256\tdX384\tdX512\tdX768\tdX1024\tdX1536\tdX2048"
              "\tvalue_rel_8_16\tvalue_rel_16_24\tvalue_rel_24_128\tvalue_rel_128_2048"
              "\tdx_rel_8_16\tdx_rel_16_24\tdx_rel_24_128\tdx_rel_128_2048"
              "\tqf_arc_value_512\tqf_arc_value_1024"
              "\tqf_arc_dx_512\tqf_arc_dx_1024\tqf_arc_dx_2048"
              "\tqf_arc_dx_4096\tqf_arc_dx_8192"
              "\tK16_density\tK24_density\tK32_density\tK64_density\tK128_density"
              "\tK256_density\tK512_density\tK1024_density\tK2048_density"
              "\n";
        const std::array<int,16> kn{{8,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048}};
        std::array<KOrderRule,16> krules{};
        for(std::size_t i=0;i<kn.size();++i)krules[i]=KOrderRule(kn[i]);
        const double D=2.0*kPi*p.rho*p.rho/3.0;
        const PrimaryFrame pf=PrimaryFrame::from(p);
        for(double x:{-0.5,0.0,0.5,0.75,0.9,0.95,0.98,0.99,0.995,0.999,0.9999}) {
            const auto rr=map(x);
            QuarticWarm qw;RootPairWarm rw;QuarticCoeffs arc_pc{};
            const auto arcs=adaptive_detail::adaptive_arc_intervals(
                rr[0],pf,&qw,&rw,&arc_pc,cell.n_crossings);
            RefValue qdiag;
            const RefArcs qarcs=certified_arcs(rr[0],pf,qdiag);
            if(arcs.kind!=ArcKind::kArcs || qarcs.kind!=ArcKind::kArcs ||
               arcs.arcs.size()!=qarcs.arcs.size() || qdiag.fail) {
                korder<<x<<'\t'<<rr[0]<<'\t'<<rr[1]<<"\t-1\tinvalid\n";
                continue;
            }
            const auto dpc0=boundary_quartic_dp(rr[0],pf);
            for(std::size_t ai=0;ai<arcs.arcs.size();++ai) {
                auto te=polish_endpoint(rr[0],arcs.arcs[ai][0],pf,6);
                auto tl=polish_endpoint(rr[0],arcs.arcs[ai][1],pf,6);
                if(!te.reliable||!tl.reliable)continue;
                if(tl.theta<=te.theta)tl.theta+=kTwoPi;
                const std::array<double,5> zero{};
                for(int chart=0;chart<2;++chart) {
                    const bool reciprocal=chart!=0;
                    const auto ap=reciprocal
                        ?arc_pair_jac_reciprocal(te.theta,tl.theta,zero,zero)
                        :arc_pair_jac(te.theta,tl.theta,zero,zero);
                    if(!ap.ok)continue;
                    const QuarticCoeffs cp=reciprocal
                        ?boundary_quartic_reciprocal(arc_pc):arc_pc;
                    const QuarticParamJac dp=reciprocal
                        ?boundary_quartic_reciprocal_dp(dpc0):dpc0;
                    const RootPairDR pair_dx=root_pair_dR(
                        RootPair{ap.m,ap.v},cp.p,dp.dp[0]);
                    if(!pair_dx.ok)continue;
                    std::array<KOrderValue,16> vals{};
                    for(std::size_t qi=0;qi<kn.size();++qi)
                        vals[qi]=eval_k_order(krules[qi],ap.m,ap.v,
                            pair_dx.dm_dR,pair_dx.dv_dR,rr[0],pf,cp,dp,reciprocal);
                    const auto qf512=qf_arc_Fh_dx(rr[0],qarcs.arcs[ai].lo,
                        qarcs.arcs[ai].hi,pf,D,512);
                    const auto qf1024=qf_arc_Fh_dx(rr[0],qarcs.arcs[ai].lo,
                        qarcs.arcs[ai].hi,pf,D,1024);
                    const auto qf2048=qf_arc_Fh_dx(rr[0],qarcs.arcs[ai].lo,
                        qarcs.arcs[ai].hi,pf,D,2048);
                    const auto qf4096=qf_arc_Fh_dx(rr[0],qarcs.arcs[ai].lo,
                        qarcs.arcs[ai].hi,pf,D,4096);
                    const auto qf8192=qf_arc_Fh_dx(rr[0],qarcs.arcs[ai].lo,
                        qarcs.arcs[ai].hi,pf,D,8192);
                    KOrderValue stats{};
                    for(const auto& vv:vals)if(vv.ok) {
                        stats.min_s2=std::min(stats.min_s2,vv.min_s2);
                        stats.min_b=std::min(stats.min_b,vv.min_b);
                        stats.max_abs_t=std::max(stats.max_abs_t,vv.max_abs_t);
                    }
                    korder<<x<<'\t'<<rr[0]<<'\t'<<rr[1]<<'\t'<<ai<<'\t'
                          <<(reciprocal?"reciprocal":"t")<<'\t'<<ap.m<<'\t'<<ap.v
                          <<'\t'<<stats.max_abs_t<<'\t'<<stats.min_s2<<'\t'<<stats.min_b;
                    for(const auto& vv:vals)korder<<'\t'<<vv.value;
                    for(const auto& vv:vals)korder<<'\t'<<vv.dx;
                    auto rel=[](double a,double b) {
                        return std::fabs(a-b)/std::max(std::fabs(b),1e-300);
                    };
                    const auto& a8=vals[0];const auto& a16=vals[1];
                    const auto& a24=vals[2];const auto& a128=vals[7];
                    const auto& a2048=vals[15];
                    korder<<'\t'<<(a8.ok&&a16.ok?rel(a8.value,a16.value):NAN)
                          <<'\t'<<(a16.ok&&a24.ok?rel(a16.value,a24.value):NAN)
                          <<'\t'<<(a24.ok&&a128.ok?rel(a24.value,a128.value):NAN)
                          <<'\t'<<(a128.ok&&a2048.ok?rel(a128.value,a2048.value):NAN)
                          <<'\t'<<(a8.ok&&a16.ok?rel(a8.dx,a16.dx):NAN)
                          <<'\t'<<(a16.ok&&a24.ok?rel(a16.dx,a24.dx):NAN)
                          <<'\t'<<(a24.ok&&a128.ok?rel(a24.dx,a128.dx):NAN)
                          <<'\t'<<(a128.ok&&a2048.ok?rel(a128.dx,a2048.dx):NAN)
                          <<'\t'<<qf512.value<<'\t'<<qf1024.value
                          <<'\t'<<qf512.dx<<'\t'<<qf1024.dx<<'\t'<<qf2048.dx
                          <<'\t'<<qf4096.dx<<'\t'<<qf8192.dx;
                    for(int idx:{1,2,3,5,7,9,11,13,15}) {
                        const auto& vv=vals[static_cast<std::size_t>(idx)];
                        korder<<'\t'<<(vv.ok?rr[1]*vv.dx/D:NAN);
                    }
                    korder<<'\n';
                }
            }
            korder.flush();
        }
        std::ofstream kint(std::string(argv[2])+".cell6_krule_integral.tsv",
                           std::ios::trunc);
        kint<<std::setprecision(17)<<"order\tcell6_gradient\twall_ms\n";
        for(int n:{16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048}) {
            const auto start=std::chrono::steady_clock::now();
            const double result=integrate_cell6_krule(n,p,cell,map);
            const auto stop=std::chrono::steady_clock::now();
            const double ms=std::chrono::duration<double,std::milli>(stop-start).count();
            kint<<n<<'\t'<<result<<'\t'<<ms<<'\n';
            kint.flush();
        }

        std::ofstream angular_out(std::string(argv[2])+".cell6_angular.tsv",
                                  std::ios::trunc);
        angular_out<<std::setprecision(17)
            <<"x\tR\tqf_n256\tqf_n512\tqf_n1024"
              "\trescue_n32\trescue_n64\trescue_n128"
              "\trescue_n256\trescue_n512"
              "\tsine_n16\tsine_n32\tsine_n64\tsine_n128\n";
        for(double x:{0.75,0.9,0.98,0.995,0.9999}) {
            const double R=map(x)[0];
            bool ok256=true,ok512=true,ok1024=true;
            const double ref256=fixed_R_dmu_dx(R,1.0,p,Rule(256),ok256);
            const double ref512=fixed_R_dmu_dx(R,1.0,p,Rule(512),ok512);
            const double ref1024=fixed_R_dmu_dx(R,1.0,p,Rule(1024),ok1024);
            bool o32=true,o64=true,o128=true,o256=true,o512=true;
            bool s16=true,s32=true,s64=true,s128=true;
            const double a32=angular_rescue_dmu_dx<32>(R,p,PrimaryFrame::from(p),cell,o32);
            const double a64=angular_rescue_dmu_dx<64>(R,p,PrimaryFrame::from(p),cell,o64);
            const double a128=angular_rescue_dmu_dx<128>(R,p,PrimaryFrame::from(p),cell,o128);
            const double a256=angular_rescue_dmu_dx<256>(R,p,PrimaryFrame::from(p),cell,o256);
            const double a512=angular_rescue_dmu_dx<512>(R,p,PrimaryFrame::from(p),cell,o512);
            const double z16=sine_rescue_dmu_dx<16>(R,p,PrimaryFrame::from(p),cell,s16);
            const double z32=sine_rescue_dmu_dx<32>(R,p,PrimaryFrame::from(p),cell,s32);
            const double z64=sine_rescue_dmu_dx<64>(R,p,PrimaryFrame::from(p),cell,s64);
            const double z128=sine_rescue_dmu_dx<128>(R,p,PrimaryFrame::from(p),cell,s128);
            angular_out<<x<<'\t'<<R<<'\t'<<ref256<<'\t'<<ref512<<'\t'
                <<ref1024<<'\t'<<a32<<'\t'<<a64<<'\t'<<a128<<'\t'
                <<a256<<'\t'<<a512<<'\t'<<z16<<'\t'<<z32<<'\t'
                <<z64<<'\t'<<z128<<'\n';
            angular_out.flush();
        }

        std::ofstream integral_out(std::string(argv[2])+".cell6_integral.tsv",
                                   std::ios::trunc);
        integral_out<<std::setprecision(17)
            <<"angular_nodes\tcell6_angular\twhole_with_cell6_replaced\twall_ms\n";
        for(int n:{32,64,128,256,512,1024,2048}) {
            double v=NAN;
            const auto start=std::chrono::steady_clock::now();
            if(n==32)v=integrate_cell6_angular<32>(p,cell,map);
            if(n==64)v=integrate_cell6_angular<64>(p,cell,map);
            if(n==128)v=integrate_cell6_angular<128>(p,cell,map);
            if(n==256)v=integrate_cell6_angular<256>(p,cell,map);
            if(n==512)v=integrate_cell6_angular<512>(p,cell,map);
            if(n==1024)v=integrate_cell6_angular<1024>(p,cell,map);
            if(n==2048)v=integrate_cell6_angular<2048>(p,cell,map);
            const auto stop=std::chrono::steady_clock::now();
            const double ms=std::chrono::duration<double,std::milli>(stop-start).count();
            integral_out<<n<<'\t'<<v<<'\t'
                        <<mapped_whole_baseline+(v-mapped_cell6_baseline)<<'\t'<<ms<<'\n';
            integral_out.flush();
        }
        std::ofstream sine_out(std::string(argv[2])+".cell6_sine_integral.tsv",
                               std::ios::trunc);
        sine_out<<std::setprecision(17)
            <<"angular_nodes\tcell6_angular_sine_map\twhole_with_cell6_replaced\n";
        for(int n:{16,32,64,128}) {
            double v=NAN;
            if(n==16)v=integrate_cell6_sine<16>(p,cell,map);
            if(n==32)v=integrate_cell6_sine<32>(p,cell,map);
            if(n==64)v=integrate_cell6_sine<64>(p,cell,map);
            if(n==128)v=integrate_cell6_sine<128>(p,cell,map);
            sine_out<<n<<'\t'<<v<<'\t'
                    <<mapped_whole_baseline+(v-mapped_cell6_baseline)<<'\n';
            sine_out.flush();
        }
    }
    return 0;
}
