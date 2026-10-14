#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <iostream>
using namespace lcbinint::holonomic;
int main(){
 int checks=0,failures=0;
 auto check=[&](bool ok,const char* msg){++checks;if(!ok){++failures;std::cerr<<msg<<'\n';}};
 for(int level=2;level<=6;++level){
  const auto& rule=radial_rule(level,true);const int n=(1<<level)-1;
  for(int degree=0;degree<=2*n-1;++degree){
   long double value=0;
   for(int k=0;k<n;++k)value+=rule.w[k]*std::pow((long double)rule.x[k],degree);
   long double exact=degree%2?0:std::exp(std::lgamma((degree+1)/2.L)+std::lgamma(1.5L)-std::lgamma((degree+4)/2.L));
   check(std::fabs(value-exact)<2e-13,"GC2 polynomial weighted moments");
  }
  for(int k=2;k<(1<<level);k+=2)
   check(rule.x[k-1]==radial_rule(level-1,true).x[k/2-1],"weighted inherited nodes are exact");
 }
 // Odd functions integrate to zero under the symmetric weight. Their
 // slopes must not force refinement of a constant even component.
 for(bool odd:{false,true}) {
  AdaptiveWorkspace w;AdaptivePanel p;p.map={0,1,false,false};p.weighted=true;w.panels.push_back(p);
  AdaptiveConfig cfg;cfg.weighted_radial=true;cfg.tol.mu_atol=1e-12;cfg.tol.mu_rtol=1e-8;
  const auto r=adaptive_detail::integrate(w,cfg,[&](double R,double J,int,const AdaptiveSample*) {
   const double x=2*R-1;
   AdaptiveSample out;out.value[0]=std::sqrt(R*(1-R))*(1+(odd?.4*x+.2*x*x*x:0))*J;return out;
  });
  check(r.value_converged && std::fabs(r.mu-kPi/8)<1e-14,"odd component preserves exact weighted integral");
  check(r.stats.node_evaluations==7,"odd slope retains minimum seven weighted nodes");
 }
 for(int degree=0;degree<=12;++degree){
  AdaptiveWorkspace w;AdaptivePanel p;p.map={0,1,false,false};p.weighted=true;w.panels.push_back(p);
  AdaptiveConfig cfg;cfg.weighted_radial=true;cfg.tol.mu_atol=1e-10;cfg.tol.mu_rtol=1e-8;
  const auto r=adaptive_detail::integrate(w,cfg,[&](double R,double J,int,const AdaptiveSample*){
   AdaptiveSample out;out.value[0]=std::sqrt(R*(1-R))*std::pow(R,degree)*J;return out;
  });
  const double exact=std::exp(std::lgamma(degree+1.5)+std::lgamma(1.5)-std::lgamma(degree+3.));
  check(r.value_converged,"weighted controller converges");
  check(std::fabs(r.mu-exact)<=cfg.tol.budget(0,exact),"weighted controller beta integral oracle");
 }
 {
  AdaptiveWorkspace w;AdaptivePanel p;p.map={0,1,false,false};p.weighted=true;w.panels.push_back(p);
  AdaptiveConfig cfg;cfg.weighted_radial=true;cfg.max_level=3;cfg.tol.mu_atol=1e-9;cfg.tol.mu_rtol=1e-6;
  auto model=[](double R,double J,int,const AdaptiveSample*){AdaptiveSample s;s.value[0]=std::sqrt(R*(1-R))*std::exp(8*R)*J;return s;};
  const auto r=adaptive_detail::integrate(w,cfg,model);
  constexpr int N=32768;long double sum=0;
  for(int i=1;i<N;++i){long double t=acosl(-1.L)*i/N,si=sinl(t),R=(1+cosl(t))/2;sum+=(i%2?4:2)*si*si*expl(8*R)/4;}
  const double exact=double(sum*acosl(-1.L)/(3*N));
  check(r.stats.splits>0,"weighted panel exercises local ordinary fold children");
  check(r.value_converged&&std::fabs(r.mu-exact)<=cfg.tol.budget(0,exact),"split preserves physical integral");
  double width=0;for(const auto& q:w.panels)if(q.active)width+=q.map(q.xr)[0]-q.map(q.xl)[0];
  check(std::fabs(width-1)<1e-14,"weighted split preserves complete radial coverage");
 }
 // A nonuniform quotient should split before exhausting the p ladder.
 // The independent theta-domain Simpson oracle includes both fold ends.
 {
  AdaptiveWorkspace w;AdaptivePanel p;p.map={0,1,false,false};p.weighted=true;w.panels.push_back(p);
  AdaptiveConfig cfg;cfg.weighted_radial=true;cfg.collect_diagnostics=true;
  cfg.tol.mu_atol=1e-10;cfg.tol.mu_rtol=1e-6;
  const auto r=adaptive_detail::integrate(w,cfg,[](double R,double J,int,const AdaptiveSample*) {
   AdaptiveSample s;s.value[0]=std::sqrt(R*(1-R))*std::exp(100*(R-1))*J;return s;
  });
  constexpr int N=32768;long double sum=0;
  for(int i=1;i<N;++i){const long double theta=acosl(-1.L)*i/N,si=sinl(theta),R=(1+cosl(theta))/2;sum+=(i%2?4:2)*si*si*expl(100*(R-1))/4;}
  const double exact=double(sum*acosl(-1.L)/(3*N));
  check(!w.panels[0].active && w.panels[0].level==4,"slow weighted quotient splits at fifteen nodes");
  check(r.value_converged && std::fabs(r.mu-exact)<=cfg.tol.budget(0,exact),"early split meets independent oracle tolerance");
  double width=0;for(const auto& q:w.panels)if(q.active)width+=q.map(q.xr)[0]-q.map(q.xl)[0];
  check(std::fabs(width-1)<1e-14,"early split preserves complete radial coverage");
 }
 {
  AdaptiveWorkspace w;AdaptivePanel p;p.map={0,1,false,false};p.weighted=true;w.panels.push_back(p);
  AdaptiveConfig cfg;cfg.weighted_radial=true;cfg.max_node_evals=22;
  cfg.tol.mu_atol=1e-10;cfg.tol.mu_rtol=1e-6;
  const auto r=adaptive_detail::integrate(w,cfg,[](double R,double J,int,const AdaptiveSample*) {
   AdaptiveSample s;s.value[0]=std::sqrt(R*(1-R))*std::exp(100*(R-1))*J;return s;
  });
  check(!r.value_converged && r.stop==AdaptiveStop::BudgetExceeded,"early split respects node budget");
  check(w.panels.size()==3 && w.panels[0].active && !w.panels[1].active && !w.panels[2].active,"partial early split retains complete parent");
  check(r.stats.node_evaluations==22,"partial early split stops after first child");
 }
 std::cout<<checks<<" checks "<<failures<<" failures\n";
 return failures?1:0;
}
