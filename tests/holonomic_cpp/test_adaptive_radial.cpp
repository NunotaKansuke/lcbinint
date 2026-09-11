#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
using namespace lcbinint::holonomic;
int failures=0,checks=0;
void check(bool x,const char* msg){++checks;if(!x){++failures;std::fprintf(stderr,"FAIL %s\n",msg);}}
int main(){
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
    // A forced four-level walk checks cache exactly without an acceptance shortcut.
    w.reset();w.panels.push_back(p);int evals=0;
    for(int l=3;l<=6;++l){auto& pp=w.panels[0];int m=1<<l,step=256/m;
        for(int k=1;k<m;++k){auto& id=pp.samples[k*step];if(id>=0)continue;id=w.samples.size();AdaptiveSample s;s.R=fejer_rule(l).x[k-1];s.value[0]=std::exp(s.R);w.samples.push_back(s);++evals;}
        check(evals==m-1,"7/15/31/63 cumulative calls");pp.level=l;adaptive_detail::estimate(pp,w,1);
        double q=0;for(int k=1;k<m;++k)q+=fejer_rule(l).w[k-1]*std::exp(fejer_rule(l).x[k-1]);check(std::fabs(pp.q[0]-q)<1e-14,"reweighted samples");}
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
    for(double u:{0.0,.5}) for(GradientPolicy policy:{GradientPolicy::None,GradientPolicy::ValueFirst}) {
        AdaptiveConfig ladder;ladder.gradient_policy=policy;ladder.with_jacobian=policy!=GradientPolicy::None;
        if(policy==GradientPolicy::ValueFirst){ladder.value_first_gradient_node_budget=0;ladder.value_first_gradient_round_budget=0;}
        ladder.tol.mu_rtol=1e-3;auto loose=epoch_adaptive(rand033,u,ladder,w);
        ladder.tol.mu_rtol=1e-4;auto tight=epoch_adaptive(rand033,u,ladder,w);
        check(loose.value_converged&&tight.value_converged,"looser event tolerance is monotone");
        check(std::isfinite(loose.mu)&&std::isfinite(tight.mu),"monotone fixture remains finite");
    }
    std::printf("%d checks %d failures\n",checks,failures);return failures?1:0;
}
