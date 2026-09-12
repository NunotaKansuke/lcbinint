// Research-only comparison of lower-tier D14 root sets against the incumbent
// final qf-backed event/topology result.  Compile with
// HOLO_D14_EVENT_CONTRACT_RESEARCH; no candidate is used by production code.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;

struct InputRow {
    int case_id=0, configuration_id=0, d_bin_index=0, epoch_index=0;
    std::string profile;
    double s=0,q=0,rho=0,x=0,y=0,time=0,u=0,X=0,reference=0;
};

static bool read_row(const std::string& line, InputRow& r) {
    if(line.empty() || line[0]=='#') return false;
    std::istringstream in(line);
    return bool(in>>r.case_id>>r.configuration_id>>r.profile>>r.d_bin_index
        >>r.epoch_index>>r.s>>r.q>>r.rho>>r.x>>r.y>>r.time>>r.u>>r.X
        >>r.reference);
}

static bool same_trajectory(const InputRow& a,const InputRow& b) {
    return a.case_id==b.case_id && a.configuration_id==b.configuration_id &&
           a.profile==b.profile && a.d_bin_index==b.d_bin_index;
}

static bool d14_event(const RadialEvent& e) {
    return e.detail.rfind("D14 ",0)==0;
}

struct CandidateEvents { std::vector<RadialEvent> events; int local_attempts=0;
    int local_successes=0; double local_max_shift=0; };

static CandidateEvents candidate_events(
    const PrimaryFrame& pf,double rmax,
    const std::vector<Cplx<__float128>>& roots,
    const std::vector<RadialEvent>& oracle_events,bool local_physical_refine,
    bool include_complex_soft=true) {
    using qf=__float128;
    CandidateEvents result;
    auto& out=result.events;
    for(const auto& e:oracle_events) if(!d14_event(e)) out.push_back(e);
    std::vector<double> soft;
    for(const auto& z:roots) {
        const qf av=fabsq(z.re),ai=fabsq(z.im);
        if(!(z.re>0)) continue;
        if(ai<=(qf)1e-8*((qf)1+av)) {
            const qf Rq=sqrtq(z.re); const double R=(double)Rq;
            if(!(R>0 && R<rmax)) continue;
            const auto probe=re_detail::probe_double_root(R,pf);
            double event_r=R,event_lo=(double)(Rq-(qf)R);
            bool physical=probe.physically_real;
            if(local_physical_refine && probe.stationary_valid) {
                ++result.local_attempts;
                adaptive_detail::EventLocation initial;
                initial.radius=R; initial.radius_lo=event_lo;
                initial.t_seed=probe.stationary_t; initial.t_seed_valid=true;
                const auto refined=adaptive_detail::refine_event(
                    R,event_lo,probe.stationary_t,pf,1e-8,initial);
                if(refined.qf_refined && std::isfinite(refined.uncertainty)) {
                    ++result.local_successes;
                    event_r=refined.radius; event_lo=refined.radius_lo;
                    result.local_max_shift=std::max(result.local_max_shift,
                        std::fabs((event_r+event_lo)-(R+(double)(Rq-(qf)R))));
                    physical=true;
                } else physical=false;
            }
            RadialEvent e{event_r,physical?"physical_real":"physical_complex",
                          physical,"D14 candidate real root"};
            e.radius_lo=event_lo; e.precision_tier=2;
            e.fold_t_seed=probe.stationary_t;
            e.fold_t_seed_valid=probe.stationary_valid &&
                                     std::isfinite(probe.stationary_t);
            out.push_back(e);
        } else {
            const double re=(double)z.re;
            if(include_complex_soft && re>0 && std::sqrt(re)<rmax) soft.push_back(re);
        }
    }
    std::sort(soft.begin(),soft.end());
    double previous=-1;
    for(double v:soft) {
        if(previous>=0 && v-previous<=1e-9) continue;
        previous=v;
        out.push_back({std::sqrt(v),"physical_complex",false,
                       "D14 candidate complex root (Re v)"});
    }
    std::sort(out.begin(),out.end(),[](const auto& a,const auto& b){
        return a.radius<b.radius;
    });
    std::vector<RadialEvent> merged;
    for(const auto& e:out) {
        const bool preserve_physical_pair=!merged.empty() &&
            merged.back().kind=="physical_real" && e.kind=="physical_real";
        if(!preserve_physical_pair && !merged.empty() &&
           merged.back().kind==e.kind &&
           std::fabs(e.radius-merged.back().radius)<1e-7) continue;
        merged.push_back(e);
    }
    result.events=std::move(merged);
    return result;
}

struct EventSummary { int positive=0,physical=0,real_soft=0,complex_soft=0; };
static EventSummary summarize_events(const std::vector<RadialEvent>& events) {
    EventSummary s;
    for(const auto& e:events) if(d14_event(e) ||
            e.detail.rfind("D14 candidate",0)==0) {
        if(e.detail.find("complex root")!=std::string::npos) ++s.complex_soft;
        else { ++s.positive; if(e.physically_real) ++s.physical;
               else ++s.real_soft; }
    }
    return s;
}

static std::vector<double> radii(const std::vector<RadialEvent>& events,
                                 int category) {
    std::vector<double> r;
    for(const auto& e:events) {
        if(!(d14_event(e)||e.detail.rfind("D14 candidate",0)==0)) continue;
        const bool complex=e.detail.find("complex root")!=std::string::npos;
        if((category==0&&!complex)||(category==1&&complex)) r.push_back(e.radius);
    }
    std::sort(r.begin(),r.end()); return r;
}

static double ordered_error(const std::vector<double>& a,
                            const std::vector<double>& b) {
    if(a.size()!=b.size()) return -1;
    double e=0; for(size_t i=0;i<a.size();++i)e=std::max(e,std::fabs(a[i]-b[i]));
    return e;
}

static int ambiguous_roots(const re_detail::D14StructQf& sc,
                           const std::vector<Cplx<__float128>>& roots) {
    using qf=__float128; int ambiguous=0;
    for(size_t i=0;i<roots.size();++i) {
        Cplx<qf> p,dp; re_detail::d14_struct_eval(sc,roots[i],p,dp);
        const qf adp=cabs(dp); qf correction=HUGE_VALQ,sep=HUGE_VALQ;
        if(adp>0 && finiteq(adp)) correction=cabs(p)/adp;
        for(size_t j=0;j<roots.size();++j) if(i!=j)
            sep=std::min(sep,cabs(roots[i]-roots[j]));
        const qf real_margin=fabsq(roots[i].im)-(qf)1e-8*((qf)1+fabsq(roots[i].re));
        if(!finiteq(correction)||!finiteq(sep)||correction*8>=sep ||
           fabsq(real_margin)<=correction*8 ||
           (roots[i].re>0 && roots[i].re<=correction*8)) ++ambiguous;
    }
    return ambiguous;
}

static D14EventContractCandidate rootwise_newton_candidate(
    const re_detail::D14StructQf& sc,
    const std::vector<__float128>& descending,
    const D14EventContractCandidate& source,int sweeps) {
    using qf=__float128;
    D14EventContractCandidate out=source;
    out.stage="presearch_newton"+std::to_string(sweeps);
    out.finite=out.roots.size()==14;
    for(int sweep=0;sweep<sweeps && out.finite;++sweep) {
        for(auto& root:out.roots) {
            Cplx<qf> p,dp; re_detail::d14_struct_eval(sc,root,p,dp);
            if(!finiteq(p.re)||!finiteq(p.im)||!finiteq(dp.re)||!finiteq(dp.im)||
               (dp.re==0 && dp.im==0)) { out.finite=false; break; }
            const auto step=p/dp;
            root=root-step;
            if(!finiteq(root.re)||!finiteq(root.im)) { out.finite=false; break; }
        }
    }
    qf scale=0,worst=0;
    for(qf c:descending) scale=std::max(scale,fabsq(c));
    if(out.finite) for(const auto& root:out.roots) {
        Cplx<qf> p,dp;re_detail::d14_struct_eval(sc,root,p,dp);
        worst=std::max(worst,cabs(p)/(scale+(qf)1e-300));
    }
    out.worst_residual=(double)worst;
    out.converged=false;
    return out;
}

// Research certificate for a finite candidate root set.  Around every
// candidate c, expand D14(c+w)=sum a_k w^k and search for a disjoint disk on
// which the linear term dominates all other terms.  Rouche's theorem then
// gives exactly one polynomial root in each disk; fourteen disjoint disks
// certify completeness without requiring the Aberth iterates themselves to
// satisfy a global step threshold.
struct RoucheCertificate {
    bool certified=false;
    int disks=0;
    double min_margin=0;
};

static RoucheCertificate rouche_root_certificate(
    const std::vector<__float128>& ascending,
    const std::vector<Cplx<__float128>>& roots) {
    using qf=__float128;
    RoucheCertificate out;
    if(ascending.size()!=15 || roots.size()!=14) return out;
    qf min_margin=HUGE_VALQ;
    for(size_t i=0;i<roots.size();++i) {
        const auto c=roots[i];
        if(!finiteq(c.re)||!finiteq(c.im)) return out;
        std::array<Cplx<qf>,15> shifted{};
        // Direct degree-14 shift is small and keeps this diagnostic simple.
        for(int k=0;k<=14;++k) {
            Cplx<qf> power{1,0};
            for(int exponent=k;exponent>=0;--exponent) {
                int choose=1;
                for(int j=1;j<=exponent;++j) choose=choose*(k-j+1)/j;
                shifted[exponent]=shifted[exponent]+
                    power*Cplx<qf>{ascending[k]*(qf)choose,0};
                power=power*c;
            }
        }
        qf sep=HUGE_VALQ;
        for(size_t j=0;j<roots.size();++j) if(i!=j)
            sep=std::min(sep,cabs(c-roots[j]));
        if(!finiteq(sep)||!(sep>0)) return out;
        const qf linear=cabs(shifted[1]);
        if(!finiteq(linear)||!(linear>0)) return out;
        qf radius=std::max((qf)16*cabs(shifted[0])/linear,
                           (qf)1e-30*((qf)1+cabs(c)));
        bool one=false;
        for(int attempt=0;attempt<24 && radius<sep/(qf)3;++attempt) {
            qf lhs=linear*radius;
            qf rhs=cabs(shifted[0]), power=radius*radius;
            for(int k=2;k<=14;++k) { rhs+=cabs(shifted[k])*power; power*=radius; }
            // Require a wide margin so binary128 rounding cannot decide a
            // near equality.  This is a research gate, not an interval proof.
            if(finiteq(lhs)&&finiteq(rhs)&&lhs>(qf)16*rhs) {
                min_margin=std::min(min_margin,lhs/(rhs+FLT128_MIN));
                one=true; break;
            }
            radius*=2;
        }
        if(!one) return out;
        ++out.disks;
    }
    out.certified=out.disks==14;
    out.min_margin=(double)min_margin;
    return out;
}

static AdaptiveResult integrate(const InputRow& row,const TopologyResult& topo,
                                double rtol) {
    const LensParams p{row.time,row.y,row.rho,1.0/row.q,row.s,true};
    AdaptiveConfig cfg; cfg.gradient_policy=GradientPolicy::None;
    cfg.with_jacobian=false; cfg.tol.mu_atol=1e-16; cfg.tol.mu_rtol=rtol;
    cfg.preserve_radial_offset=true;
    AdaptiveWorkspace workspace;
    return flux_adaptive_integrate(p,row.u,topo,cfg,workspace);
}

int main(int argc,char** argv) {
    if(argc<3) { std::cerr<<"usage: bench_d14_event_contract INPUT OUT.tsv [case=-1]\n"; return 2; }
    const int case_filter=argc>3?std::atoi(argv[3]):-1;
    std::ifstream input(argv[1]); std::ofstream out(argv[2]);
    if(!input||!out) return 2;
    std::vector<InputRow> rows; std::string line;
    while(std::getline(input,line)){InputRow r;if(read_row(line,r)&&
        (case_filter<0||r.case_id==case_filter))rows.push_back(r);}
    std::stable_sort(rows.begin(),rows.end(),[](const auto&a,const auto&b){
        if(a.case_id!=b.case_id)return a.case_id<b.case_id;
        if(a.configuration_id!=b.configuration_id)return a.configuration_id<b.configuration_id;
        if(a.profile!=b.profile)return a.profile<b.profile;
        if(a.d_bin_index!=b.d_bin_index)return a.d_bin_index<b.d_bin_index;
        return a.epoch_index<b.epoch_index;});
    out<<std::setprecision(17)
       <<"case_id configuration_id profile d_bin_index epoch_index lane stage "
         "candidate_converged candidate_finite scalar_certificate residual ambiguous_roots "
         "oracle_tier candidate_positive oracle_positive candidate_physical oracle_physical "
         "candidate_real_soft oracle_real_soft candidate_complex_soft oracle_complex_soft "
         "positive_radius_maxdiff soft_radius_maxdiff candidate_cells oracle_cells "
         "candidate_topology_status oracle_topology_status rtol candidate_stop oracle_stop "
         "candidate_value_converged oracle_value_converged candidate_mu oracle_mu mu_absdiff "
         "local_attempts local_successes local_max_shift event_build_ms "
         "positive_certificate positive_count positive_certificate_ms positive_reason "
         "rouche_certificate rouche_disks rouche_min_margin rouche_ms "
         "interval_rouche_certificate interval_rouche_disks interval_rouche_ratio interval_rouche_ms\n";
    for(bool warm:{false,true}) {
        std::vector<Cplx<__float128>> previous;
        for(size_t k=0;k<rows.size();++k) {
            if(k==0||!same_trajectory(rows[k-1],rows[k])) previous.clear();
            const auto& row=rows[k];
            const LensParams p{row.time,row.y,row.rho,1.0/row.q,row.s,true};
            const auto pf=PrimaryFrame::from(p);
            D14EventContractCapture capture;
            TopologyResult oracle;
            std::vector<Cplx<__float128>> current;
            { D14EventContractCaptureScope scope(capture);
              oracle=classify_cells(pf,warm&&!previous.empty()?&previous:nullptr,
                                    &current,true); }
            auto sc=re_detail::d14_struct_build((__float128)pf.a,(__float128)pf.m0,
                (__float128)pf.X,(__float128)pf.Y,(__float128)pf.rho);
            auto descv=re_detail::d14_expanded_from_struct(sc);
            std::reverse(descv.begin(),descv.end());
            const auto os=summarize_events(oracle.events);
            auto candidates=capture.candidates;
            for(const auto& candidate:capture.candidates) {
                if(candidate.stage!="presearch" || !candidate.finite) continue;
                for(int sweeps:{1,2,4})
                    candidates.push_back(
                        rootwise_newton_candidate(sc,descv,candidate,sweeps));
                break;
            }
            D14EventContractCandidate oracle_candidate;
            oracle_candidate.stage="oracle";
            oracle_candidate.roots=capture.oracle_roots;
            oracle_candidate.finite=oracle_candidate.roots.size()==14;
            oracle_candidate.converged=true;
            candidates.push_back(std::move(oracle_candidate));
            for(const auto& c:candidates) {
                if(!c.finite) continue;
              PositiveD14Result positive;
              double positive_ms=0;
              RoucheCertificate rouche;
              double rouche_ms=0;
              re_detail::D14RoucheCertificate interval_rouche;
              double interval_rouche_ms=0;
              if(c.stage=="qf_warm" && !c.converged) {
                  const auto begin=std::chrono::steady_clock::now();
                  positive=positive_d14_roots(pf,capture.r_max);
                  positive_ms=std::chrono::duration<double,std::milli>(
                      std::chrono::steady_clock::now()-begin).count();
                  auto ascending=descv; std::reverse(ascending.begin(),ascending.end());
                  const auto rouche_begin=std::chrono::steady_clock::now();
                  rouche=rouche_root_certificate(ascending,c.roots);
                  rouche_ms=std::chrono::duration<double,std::milli>(
                      std::chrono::steady_clock::now()-rouche_begin).count();
                  const auto interval_begin=std::chrono::steady_clock::now();
                  interval_rouche=re_detail::d14_rouche_certificate(pf,c.roots);
                  interval_rouche_ms=std::chrono::duration<double,std::milli>(
                      std::chrono::steady_clock::now()-interval_begin).count();
              }
              for(int variant=0;variant<3;++variant) {
                if(c.stage=="oracle" && variant!=2) continue;
                const bool local_refine=variant==1;
                const bool include_complex_soft=variant!=2;
                const auto event_start=std::chrono::steady_clock::now();
                auto ce=candidate_events(pf,capture.r_max,c.roots,oracle.events,
                                         local_refine,include_complex_soft);
                const double event_build_ms=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-event_start).count();
                auto cs=summarize_events(ce.events);
                auto candidate=classify_event_cells(pf,std::move(ce.events),capture.r_max);
                __float128 cert_res=0;int cert_reason=0;
                bool cert=re_detail::d14_scalar_certificate(&sc,descv,c.roots,
                                                            &cert_res,&cert_reason);
                const double pd=ordered_error(radii(candidate.events,0),radii(oracle.events,0));
                const double sd=ordered_error(radii(candidate.events,1),radii(oracle.events,1));
                for(double rtol:{1e-3,1e-4}) {
                    const auto cv=integrate(row,candidate,rtol),ov=integrate(row,oracle,rtol);
                    out<<row.case_id<<' '<<row.configuration_id<<' '<<row.profile<<' '
                       <<row.d_bin_index<<' '<<row.epoch_index<<' '<<(warm?"warm":"cold")<<' '
                       <<c.stage<<(local_refine?"_local":
                                      (!include_complex_soft?"_nosoft":""))<<' '
                       <<c.converged<<' '<<c.finite<<' '<<cert<<' '
                       <<c.worst_residual<<' '<<ambiguous_roots(sc,c.roots)<<' '
                       <<capture.oracle_tier<<' '<<cs.positive<<' '<<os.positive<<' '
                       <<cs.physical<<' '<<os.physical<<' '<<cs.real_soft<<' '<<os.real_soft<<' '
                       <<cs.complex_soft<<' '<<os.complex_soft<<' '<<pd<<' '<<sd<<' '
                       <<candidate.cells.size()<<' '<<oracle.cells.size()<<' '
                       <<int(candidate.status)<<' '<<int(oracle.status)<<' '<<rtol<<' '
                       <<int(cv.stop)<<' '<<int(ov.stop)<<' '<<cv.value_converged<<' '
                       <<ov.value_converged<<' '<<cv.mu<<' '<<ov.mu<<' '
                       <<std::fabs(cv.mu-ov.mu)<<' '<<ce.local_attempts<<' '
                       <<ce.local_successes<<' '<<ce.local_max_shift<<' '
                       <<event_build_ms<<' '
                       <<int(positive.assurance==PositiveRootAssurance::PositiveRealCertified)
                       <<' '<<positive.root_count<<' '<<positive_ms<<' '
                       <<positive.stats.reason<<' '<<rouche.certified<<' '
                       <<rouche.disks<<' '<<rouche.min_margin<<' '<<rouche_ms<<' '
                       <<interval_rouche.certified<<' '<<interval_rouche.isolated_disks
                       <<' '<<(double)interval_rouche.minimum_ratio<<' '
                       <<interval_rouche_ms<<'\n';
                }
              }
            }
            if(warm) previous=std::move(current);
        }
    }
}
