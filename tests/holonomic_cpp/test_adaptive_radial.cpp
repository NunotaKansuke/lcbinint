#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
using namespace lcbinint::holonomic;
int failures=0,checks=0;
void check(bool x,const char* msg){++checks;if(!x){++failures;std::fprintf(stderr,"FAIL %s\n",msg);}}
int main(){
    // The local DD/qf fold kernel must reproduce the incumbent explicit
    // double quartic and its R derivative before it is used to replace the
    // generic polynomial-family fallback.
    LensParams fold_probe_lens{.31,-.17,.043,.27,.83,false};
    const auto fold_pf=PrimaryFrame::from(fold_probe_lens);
    for(double R:{.07,.41,1.3}) for(double t:{-.8,.2,2.1}) {
        const auto pc=boundary_quartic(R,fold_pf);
        const auto pr=boundary_quartic_dR(R,fold_pf);
        const auto g=local_fold_quantities<double>(R,t,fold_pf);
        const double scale=1+std::fabs(g.P)+std::fabs(g.Pt)+std::fabs(g.PR)+
                           std::fabs(g.Ptt)+std::fabs(g.Ptr);
        check(std::fabs(g.P-adaptive_detail::eval_poly5(pc.p,t))<2e-13*scale,
              "local fold P parity");
        check(std::fabs(g.Pt-adaptive_detail::eval_poly5_derivative(pc.p,t))<2e-13*scale,
              "local fold Pt parity");
        check(std::fabs(g.PR-adaptive_detail::eval_poly5(pr.p,t))<2e-13*scale,
              "local fold PR parity");
        const double ptt=2*pc.p[2]+t*(6*pc.p[3]+t*12*pc.p[4]);
        const double ptr=pr.p[1]+t*(2*pr.p[2]+t*(3*pr.p[3]+t*4*pr.p[4]));
        check(std::fabs(g.Ptt-ptt)<2e-13*scale,"local fold Ptt parity");
        check(std::fabs(g.Ptr-ptr)<2e-13*scale,"local fold Ptr parity");
    }
    AdaptiveTolerance tol;tol.mu_atol=.1;tol.mu_rtol=.2;
    check(tol.budget(0,1)==.2,"max budget, not additive");
    check(tol.budget(0,0)==.1,"absolute tolerance near zero");
    check(tol.budget(0,-1)==.2,"relative tolerance uses absolute value");
    check(tol.budget(0,.5)==.1,"equal absolute and relative budgets are not added");
    for(int j=1;j<=5;++j) {
        tol.grad_atol[j-1]=.1;tol.grad_rtol[j-1]=.2;
        check(tol.budget(j,-1)==.2,"Jacobian max budget");
        check(tol.budget(j,0)==.1,"zero Jacobian uses absolute budget");
    }
    for(int l=2;l<=8;++l){const auto& r=fejer_rule(l);double sum=0;for(double a:r.w){check(a>0,"positive weight");sum+=a;}check(std::fabs(sum-2)<2e-14,"weight sum");
        for(int d=0;d<int(r.x.size());++d){double q=0;for(size_t k=0;k<r.x.size();++k)q+=r.w[k]*std::cos(d*std::acos(r.x[k]));double exact=d%2?0:2.0/(1.-d*d);check(std::fabs(q-exact)<2e-12,"Chebyshev exactness");}
        for(int k=1;k<(1<<(l-1));++k)check(r.x[2*k-1]==fejer_rule(l-1).x[k-1],"bitwise nested");}
    AdaptiveWorkspace w;AdaptivePanel p;p.map={0,1,true,false};w.panels.push_back(p);
    AdaptiveConfig cfg;cfg.with_jacobian=true;cfg.tol.mu_atol=1e-12;cfg.tol.mu_rtol=1e-10;
    auto model=[](double R,double J,int,const AdaptiveSample*){AdaptiveSample s;s.value[0]=std::sqrt(R)*J;s.value[1]=J/std::sqrt(R);return s;};
    auto result=adaptive_detail::integrate(w,cfg,model);
    check(result.stop==AdaptiveStop::Converged,"fold model converges");check(std::fabs(result.mu-2./3)<1e-12,"fold value");check(std::fabs(result.grad_mu[0]-2)<1e-12,"fold derivative");
    check(result.stats.unique_nodes==w.samples.size(),"only new nodes counted");
    AdaptiveConfig value_first;value_first.gradient_policy=GradientPolicy::ValueFirst;
    value_first.value_first_gradient_node_budget=0;value_first.value_first_gradient_round_budget=0;
    value_first.tol.mu_atol=1e-12;value_first.tol.mu_rtol=1e-10;
    w.reset();AdaptivePanel value_panel;value_panel.map={0,1,false,false};w.panels.push_back(value_panel);
    auto finite_unverified=[](double R,double J,int,const AdaptiveSample*){AdaptiveSample s;s.value[0]=J;s.value[1]=J*std::sin(100*R);return s;};
    auto value_first_result=adaptive_detail::integrate(w,value_first,finite_unverified);
    check(value_first_result.value_converged&&value_first_result.stop==AdaptiveStop::Converged,"ValueFirst keeps converged value");
    check(value_first_result.grad_quality[0]==GradientQuality::FiniteUncertified&&
          value_first_result.grad_reason[0]==GradientReason::BudgetExceeded,"ValueFirst separates finite gradient quality");
    check(value_first_result.numerical_status==Status::OK,"finite uncertified gradient does not invalidate value");
    w.reset();w.panels.push_back(value_panel);
    auto invalid_gradient=[](double,double J,int,const AdaptiveSample*){AdaptiveSample s;s.value[0]=J;s.value[1]=NAN;return s;};
    auto invalid_gradient_result=adaptive_detail::integrate(w,value_first,invalid_gradient);
    check(invalid_gradient_result.value_converged&&invalid_gradient_result.stop==AdaptiveStop::Nonfinite,"invalid gradient preserves value but fails status");
    check(invalid_gradient_result.grad_quality[0]==GradientQuality::Invalid&&
          invalid_gradient_result.grad_reason[0]==GradientReason::Nonfinite,"invalid gradient is not uncertified");
    w.reset();w.panels.push_back(value_panel);
    auto one_invalid_gradient=[](double,double J,int,const AdaptiveSample*){AdaptiveSample s;s.value[0]=J;s.value[1]=NAN;s.value[2]=J;return s;};
    auto one_invalid_result=adaptive_detail::integrate(w,value_first,one_invalid_gradient);
    check(one_invalid_result.grad_quality[0]==GradientQuality::Invalid&&
          one_invalid_result.grad_quality[1]!=GradientQuality::Invalid,
          "gradient invalidity is tracked per component");
    // A fully converged gradient must terminate at the initial rule, rather
    // than refining forever because the all-components AND starts at false.
    {
        AdaptiveWorkspace aw;AdaptivePanel ap;ap.map={0,1,false,false};aw.panels.push_back(ap);
        AdaptiveConfig ac;ac.gradient_policy=GradientPolicy::Strict;
        auto ar=adaptive_detail::integrate(aw,ac,[](double,double J,int,const AdaptiveSample*){
            AdaptiveSample s;for(int j=0;j<6;++j)s.value[j]=(j+1)*J;return s;
        });
        check(ar.stop==AdaptiveStop::Converged&&ar.stats.unique_nodes==7,
              "all converged gradients stop at initial rule");
        check(ar.stats.first_gradient_contract_nodes==7,
              "gradient convergence records first successful mesh");
    }
    // The primal integral is constant but its parameter derivative has a
    // sharp boundary layer (f(R,p)=1+p*exp(20R), evaluated at p=0).
    // Interrupt a split after one child: the complete parent must survive.
    {
        auto trial=[&](int rounds,size_t budget,bool fail_right){
            AdaptiveWorkspace aw;AdaptivePanel ap;ap.map={0,1,false,false};aw.panels.push_back(ap);
            AdaptiveConfig ac;ac.gradient_policy=GradientPolicy::ValueFirst;
            ac.max_level=3;ac.value_first_gradient_round_budget=rounds;
            ac.value_first_gradient_node_budget=budget;int calls=0;
            auto ar=adaptive_detail::integrate(aw,ac,[&](double R,double J,int,const AdaptiveSample*){
                AdaptiveSample s;s.value[0]=J;s.value[1]=J*std::exp(20*R);
                if(fail_right&&++calls==15)s.reliable=false;
                return s;
            });
            double covered=0;bool complete=true;
            for(const auto& panel:aw.panels)if(panel.active){
                covered+=(panel.xr-panel.xl)/2;complete=complete&&panel.level>=3;
            }
            check(covered==1&&complete,"interrupted split retains full domain coverage");
            return ar;
        };
        const auto before=trial(0,4096,false);
        for(const auto& ar:{trial(1,4096,false),trial(4,7,false)}){
            check(ar.mu==before.mu&&ar.grad_mu[0]==before.grad_mu[0],
                  "budget interruption preserves complete primal and derivative");
            check(ar.grad_quality[0]==GradientQuality::FiniteUncertified,
                  "budget interruption retains uncertified complete derivative");
        }
        const auto bad=trial(4,4096,true);
        check(bad.grad_quality[0]==GradientQuality::Invalid,
              "split rollback must not hide numerical invalidity");
    }
    // A forced four-level walk checks cache exactly without an acceptance shortcut.
    w.reset();w.panels.push_back(p);int evals=0;
    for(int l=3;l<=6;++l){auto& pp=w.panels[0];int m=1<<l,step=256/m;
        for(int k=1;k<m;++k){auto& id=pp.samples[k*step];if(id>=0)continue;id=w.samples.size();AdaptiveSample s;s.R=fejer_rule(l).x[k-1];s.value[0]=std::exp(s.R);w.samples.push_back(s);++evals;}
        check(evals==m-1,"7/15/31/63 cumulative calls");pp.level=l;adaptive_detail::estimate(pp,w,1);
        double q=0;for(int k=1;k<m;++k)q+=fejer_rule(l).w[k-1]*std::exp(fejer_rule(l).x[k-1]);check(std::fabs(pp.q[0]-q)<1e-14,"reweighted samples");}
    {
        auto hybrid_panel=w.panels[0],weighted_panel=w.panels[0];
        AdaptiveConfig hybrid_cfg,weighted_cfg;
        weighted_cfg.radial_error_estimator=RadialErrorEstimator::WeightedDetail;
        adaptive_detail::estimate(hybrid_panel,w,1,hybrid_cfg);
        adaptive_detail::estimate(weighted_panel,w,1,weighted_cfg);
        check(hybrid_panel.radial[0]>=
                  hybrid_cfg.weighted_detail_floor_fraction*weighted_panel.radial[0],
              "hybrid estimator retains weighted-detail safety floor");
        check(hybrid_panel.radial[0]<=weighted_panel.radial[0],
              "hybrid estimator never exceeds conservative detail");
    }
    cfg.require_bound=true;w.reset();w.panels.push_back(p);result=adaptive_detail::integrate(w,cfg,model);check(result.stop==AdaptiveStop::BoundUnavailable,"no false rigorous bound");
    cfg.require_bound=false;cfg.max_node_evals=6;result=adaptive_detail::integrate(w,cfg,model);check(result.stop==AdaptiveStop::BudgetExceeded&&!result.value_converged,"budget is not success");
    cfg.max_node_evals=32768;w.reset();w.panels.push_back(p);result=adaptive_detail::integrate(w,cfg,[](double,double,int,const AdaptiveSample*){AdaptiveSample s;s.value[0]=NAN;return s;});check(result.stop==AdaptiveStop::Nonfinite,"nonfinite");
    // Exercise the controller itself, not only the node table.
    cfg=AdaptiveConfig{};cfg.tol.mu_atol=1e-12;cfg.tol.mu_rtol=1e-10;
    auto exponential=[](double R,double J,int,const AdaptiveSample*){AdaptiveSample s;s.value[0]=std::exp(6*R)*J;return s;};
    w.reset();w.panels.push_back(p);auto cached=adaptive_detail::integrate(w,cfg,exponential);
    check(cached.stop==AdaptiveStop::Converged,"cached refinement converges");
    check(cached.stats.reused_nodes>0&&cached.stats.node_evaluations==cached.stats.unique_nodes,"only new physical calls");
    cfg.reuse_samples=false;w.reset();w.panels.push_back(p);auto uncached=adaptive_detail::integrate(w,cfg,exponential);
    check(std::fabs(cached.mu-uncached.mu)<1e-12,"cached and recomputed parity");
    check(uncached.stats.node_evaluations>uncached.stats.unique_nodes,"no-cache control really recomputes");
    cfg.reuse_samples=true;cfg.max_level=3;cfg.tol.mu_rtol=1e-5;
    w.reset();w.panels.push_back(p);auto split=adaptive_detail::integrate(w,cfg,exponential);
    check(split.stop==AdaptiveStop::Converged&&split.stats.splits>0,"h split converges");
    check(std::fabs(split.mu-(std::exp(6.)-1)/6)<1e-4,"h split analytic reference");
    check(split.stats.root_anchors>0,"h children keep parent anchors");
    cfg=AdaptiveConfig{};cfg.tol.mu_atol=.1;cfg.tol.mu_rtol=.1;
    w.reset();AdaptivePanel affine;affine.map={0,1,false,false};w.panels.push_back(affine);
    auto boundary=adaptive_detail::integrate(w,cfg,[](double,double J,int,const AdaptiveSample*){AdaptiveSample s;s.value[0]=J;s.inner[0]=.15*J;return s;});
    check(boundary.stop==AdaptiveStop::InnerAccuracyLimited&&!boundary.value_converged,"max budget rejects where additive would pass");
    cfg.max_panels=0;w.reset();w.panels.push_back(p);result=adaptive_detail::integrate(w,cfg,model);
    check(result.stop==AdaptiveStop::BudgetExceeded,"initial panel budget");
    cfg=AdaptiveConfig{};
    // A physical fold retained inside a merged EMPTY cell must become a
    // real cut; midpoint classification alone cannot certify the whole cell.
    LensParams merged_lens{-.41048657797268234,-.08583523821995483,.06874482135478452,.00011854461271149655,7.635814078934241,false};
    auto frame=PrimaryFrame::from(merged_lens);auto topology=classify_cells(frame);std::vector<CellPlan> restored;
    check(adaptive_detail::restore_physical_cuts(topology,frame,restored),"restore retained physical event");
    check(restored.size()>topology.cells.size(),"split merged cell");
    for(const auto& cell:restored)for(const auto& e:topology.events)
        if(e.physically_real&&e.kind=="physical_real")check(!(e.radius>cell.r_lo&&e.radius<cell.r_hi),"no retained physical event inside a panel");
    auto missing=topology;missing.cells.erase(missing.cells.begin());restored.clear();
    check(!adaptive_detail::restore_physical_cuts(missing,frame,restored),"skipped cell cannot silently contribute zero");
    // Physical regression: saved uniform polar self-convergence at 100/200/400
    // is 12363.35792768 / 12363.39696885 / 12363.40159240, NOT exact truth.
    LensParams lens{.00027748104958798373,.000043678813767388989,8.1133264861482082e-5,1/.00017213162835819022,1.6869965260211353,true};
    cfg.with_jacobian=false;cfg.tol.mu_rtol=1e-6;
    result=epoch_value_adaptive(lens,0,cfg,w);
    check(result.value_converged,"high magnification uniform converges");
    check(std::fabs(result.mu-12363.401592396434)<.01,"independent polar convergence interval");
    check(result.stats.unique_nodes<1024,"fold map limits work on hard fixture");
    double a=result.mu;cfg.tol.mu_rtol=1e-8;result=epoch_value_adaptive(lens,0,cfg,w);
    check(std::fabs(a-result.mu)<.01,"tolerance ladder stability");
    cfg=AdaptiveConfig{};auto invalid=lens;invalid.rho=INFINITY;
    result=epoch_value_adaptive(invalid,0,cfg,w);check(result.stop==AdaptiveStop::InvalidConfig,"reject invalid input before topology");
    LensParams origin{0,0,.1,.5,1,false};
    result=epoch_value_adaptive(origin,0,cfg,w);check(!result.value_converged,"retain near-origin fail-closed policy");
    // Regression fixture for the Phase 9.1 non-monotone event ladder: the
    // loose value tolerance must not reject the event anchor that the tighter
    // tolerance accepts.  Test both requested policies and both u values.
    LensParams rand033{.14772949073990987,.0025702291249036745,.022382854740423824,
                       .24663546751963286,.4390801625936331,false};
    // The D14/topology hand-off must make both local tiers independently
    // usable.  Exercise the retained qf radius remainder and stationary seed
    // directly, including the explicit qf fallback kernel; neither path may
    // rebuild the generic PolyFamilyR.
    const auto rand_pf=PrimaryFrame::from(rand033);
    const auto production_topology=classify_cells(rand_pf);
    const auto rand_topology=classify_cells(rand_pf,nullptr,nullptr,true);
    check(production_topology.status==rand_topology.status&&
          production_topology.cells.size()==rand_topology.cells.size(),
          "adaptive metadata keeps production topology");
    check(production_topology.events.size()==rand_topology.events.size(),
          "adaptive metadata keeps production event count");
    for(size_t i=0;i<production_topology.events.size()&&
                    i<rand_topology.events.size();++i) {
        const auto& a=production_topology.events[i];
        const auto& b=rand_topology.events[i];
        check(a.kind==b.kind&&a.physically_real==b.physically_real&&
              std::fabs(a.radius-b.radius)<=2e-14*(1+std::fabs(a.radius)),
              "adaptive metadata keeps production event parity");
    }
    size_t physical_events=0;
    for(const auto& event:rand_topology.events) {
        if(!event.physically_real||event.kind!="physical_real")continue;
        ++physical_events;
        check(event.fold_t_seed_valid,"topology retains physical stationary seed");
        auto local=adaptive_detail::topology_event_estimate(event,rand_pf,1e-4);
        check(local.dd_reason==EventDecisionReason::DDAccepted&&
              !local.needs_qf,"topology DD coupled fold correction");
        auto qf=adaptive_detail::refine_event(event.radius,event.radius_lo,
                                               event.fold_t_seed,rand_pf,1e-4,local);
        check(qf.qf_reason==EventDecisionReason::QfRefined&&!qf.needs_qf&&
              std::isfinite(qf.radius)&&std::isfinite(qf.uncertainty),
              "direct qf coupled fold correction");
    }
    check(physical_events>0,"physical event metadata fixture");
    for(double u:{0.0,.5}) for(GradientPolicy policy:{GradientPolicy::None,GradientPolicy::ValueFirst}) {
        AdaptiveConfig ladder;ladder.gradient_policy=policy;ladder.with_jacobian=policy!=GradientPolicy::None;
        if(policy==GradientPolicy::ValueFirst){ladder.value_first_gradient_node_budget=0;ladder.value_first_gradient_round_budget=0;}
        ladder.tol.mu_rtol=1e-3;auto loose=epoch_adaptive(rand033,u,ladder,w);
        ladder.tol.mu_rtol=1e-4;auto tight=epoch_adaptive(rand033,u,ladder,w);
        check(loose.value_converged&&tight.value_converged,"looser event tolerance is monotone");
        check(std::isfinite(loose.mu)&&std::isfinite(tight.mu),"monotone fixture remains finite");
    }
    // Adaptive full-circle cells use their own nested periodic evaluator;
    // compare its equation evaluation with the incumbent fixed 256-node
    // diagnostic for a deliberately smooth all-positive circle.  The fixed
    // helper is an oracle here only; the adaptive path must remain reliable.
    LensParams full_lens{.31,.17,100.0,.5,.83,false};
    const auto full_pf=PrimaryFrame::from(full_lens);
    const double full_R=1.2,full_u=.5;
    AdaptiveConfig full_cfg;full_cfg.tol.mu_rtol=1e-10;
    const auto full_adaptive=adaptive_detail::mapped_full_circle(
        full_R,1.0,full_lens,full_u,full_pf,false,&full_cfg);
    const auto full_fixed=full_circle_terms(full_R,full_pf);
    const double full_D=kPi*full_lens.rho*full_lens.rho*(1-full_u/3);
    const double full_expected=((1-full_u)*full_fixed.f0+
                                full_u*full_fixed.fh)/full_D;
    check(full_adaptive.reliable&&
          full_adaptive.reject_reason==AdaptiveSampleRejectReason::None,
          "adaptive full-circle evaluator is reliable");
    check(std::fabs(full_adaptive.value[0]-full_expected)<1e-12,
          "adaptive full-circle value matches fixed diagnostic");
    // The full-circle derivative lane is checked against an independent
    // central difference of the physical user parameters.  This exercises
    // the internal->user chain rule and the rho normalization without using
    // the incumbent derivative implementation as its oracle.
    full_lens.barycentric=true;
    const auto full_value=[&](LensParams pp) {
        const auto ppf=PrimaryFrame::from(pp);
        return adaptive_detail::mapped_full_circle(
            full_R,1.0,pp,full_u,ppf,false,&full_cfg).value[0];
    };
    const std::array<double,5> full_params{
        full_lens.xs,full_lens.ys,full_lens.rho,full_lens.q,full_lens.a};
    for(int j=0;j<5;++j) {
        const double h=2e-6*std::max(1.0,std::fabs(full_params[j]));
        LensParams plus=full_lens,minus=full_lens;
        auto set=[&](LensParams& pp,double value) {
            if(j==0)pp.xs=value; else if(j==1)pp.ys=value;
            else if(j==2)pp.rho=value; else if(j==3)pp.q=value;
            else pp.a=value;
        };
        set(plus,full_params[j]+h);set(minus,full_params[j]-h);
        const double fd=(full_value(plus)-full_value(minus))/(2*h);
        const auto deriv=adaptive_detail::mapped_full_circle(
            full_R,1.0,full_lens,full_u,PrimaryFrame::from(full_lens),true,
            &full_cfg);
        check(std::fabs(deriv.value[j+1]-fd)<
                  2e-6*std::max(1.0,std::fabs(fd)),
              "adaptive full-circle derivative chain rule");
    }
    // Regression fixtures for the two residual Phase-9 failures.  c9 has
    // two distinct physical folds closer than the historical fixed-route
    // merge tolerance; c92 has a direct-chart Aberth false real root that
    // adaptive_sturm_thetas must repair without an angular grid.
    LensParams c9{1.1422819920252716,.09455346559252209,
                  5.470597280025246e-05,992.0790840775868,
                  .57986328980431667,true};
    const auto c9_pf=PrimaryFrame::from(c9);
    const auto c9_topology=classify_cells(c9_pf,nullptr,nullptr,true);
    std::vector<double> c9_folds;
    for(const auto& e:c9_topology.events)
        if(e.physically_real&&e.kind=="physical_real")c9_folds.push_back(e.radius);
    std::sort(c9_folds.begin(),c9_folds.end());
    bool close_pair=false;
    for(size_t i=1;i<c9_folds.size();++i)
        if(c9_folds[i]-c9_folds[i-1]>9e-8&&
           c9_folds[i]-c9_folds[i-1]<1.1e-7)close_pair=true;
    check(close_pair,"adaptive metadata retains close physical folds");
    bool c9_four_cell=false;
    const CellPlan* c9_cell=nullptr;
    const double c9_R=.015950887178690351;
    for(const auto& c:c9_topology.cells)
        if(c9_R>c.r_lo&&c9_R<c.r_hi){c9_cell=&c;c9_four_cell|=c.n_crossings==4;}
    check(c9_four_cell&&c9_cell,"close-fold band has a four-crossing cell");
    if(c9_cell) {
        AdaptiveConfig c9_cfg;c9_cfg.tol.mu_rtol=1e-4;
        const auto sample=adaptive_detail::mapped_radius(
            c9_R,1.0,c9,.5,c9_pf,*c9_cell,false,false,nullptr,&c9_cfg);
        check(sample.reliable,"close-fold adaptive sample remains usable");
    }
    LensParams c92{.016817436700993973,-1.7129651189458623e-05,
                   3.1430126951461966e-05,218.24104917640977,
                   3.8691964055932191,true};
    const auto c92_pf=PrimaryFrame::from(c92);
    const auto c92_topology=classify_cells(c92_pf,nullptr,nullptr,true);
    const double c92_R=4.6917422948319514;
    const CellPlan* c92_cell=nullptr;
    for(const auto& c:c92_topology.cells)
        if(c92_R>c.r_lo&&c92_R<c.r_hi){c92_cell=&c;break;}
    check(c92_cell,"Sturm repair fixture is inside a classified cell");
    if(c92_cell) {
        AdaptiveConfig c92_cfg;c92_cfg.tol.mu_rtol=1e-4;
        const auto sample=adaptive_detail::mapped_radius(
            c92_R,1.0,c92,.5,c92_pf,*c92_cell,false,false,nullptr,&c92_cfg);
        check(sample.reliable&&sample.reject_reason==AdaptiveSampleRejectReason::None,
              "adaptive Sturm repair resolves false odd root set");
    }
    std::printf("%d checks %d failures\n",checks,failures);return failures?1:0;
}
