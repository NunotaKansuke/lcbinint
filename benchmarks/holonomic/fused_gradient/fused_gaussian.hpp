#pragma once
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>
namespace lcbinint::holonomic::fused_gradient {
struct GaussianRule {std::vector<double> x,w;};
inline GaussianRule make_gaussian_rule(int n){
 GaussianRule rule;rule.x.resize(n);rule.w.resize(n);
 constexpr long double pi=3.1415926535897932384626433832795029L;
 for(int i=0;i<(n+1)/2;++i){
  long double z=std::cos(pi*(i+.75L)/(n+.5L)),dp=0;
  for(int iter=0;iter<32;++iter){
   long double p=1,old=0;
   for(int j=1;j<=n;++j){const auto next=((2*j-1)*z*p-(j-1)*old)/j;old=p;p=next;}
   dp=n*(z*p-old)/(z*z-1);const auto step=p/dp;z-=step;
   if(std::fabs(step)<4*std::numeric_limits<long double>::epsilon())break;
  }
  const double w=double(2/((1-z*z)*dp*dp));
  rule.x[i]=double(z);rule.x[n-i-1]=double(-z);rule.w[i]=rule.w[n-i-1]=w;
 }
 return rule;
}
// Each immutable table is initialized once, with C++ thread-safe local statics.
template<int N> inline const GaussianRule& fixed_gaussian_rule() {
 static const GaussianRule rule=make_gaussian_rule(N);return rule;
}
inline const GaussianRule& gaussian_rule(int n) {
 switch(n) {
 case 14:return fixed_gaussian_rule<14>();
 case 30:return fixed_gaussian_rule<30>();
 case 62:return fixed_gaussian_rule<62>();
 case 126:return fixed_gaussian_rule<126>();
 case 254:return fixed_gaussian_rule<254>();
 case 510:return fixed_gaussian_rule<510>();
 default:throw std::invalid_argument("unsupported deferred Gaussian order");
 }
}
// Only value-driven h children defer their tangents. Unsplit panels retain
// the original fused estimate. On accepted children, 2*(2^level-1) Gauss
// points integrate polynomials through degree 4*(2^level-1)-1. No derivative
// error is estimated or used to choose the order or the accepted value mesh.
template<class Eval>
inline AdaptiveResult gaussian_replay(AdaptiveWorkspace&workspace,const AdaptiveConfig&cfg,Eval&&eval,std::size_t primal_nodes){
 AdaptiveResult result;std::array<adaptive_detail::Sum,5> sum;
 bool valid=true;AdaptiveStop failure=AdaptiveStop::TopologyUnresolved;
 for(const auto&panel:workspace.panels)if(panel.active){
  if(panel.depth==0){for(int j=0;j<5;++j)sum[j].add(panel.q[j+1]);continue;}

  const int level=std::max(cfg.initial_level,panel.level);
  const auto&rule=gaussian_rule(((1<<level)-1)*2);
  const double half=.5*(panel.xr-panel.xl),mid=.5*(panel.xr+panel.xl);
  std::vector<std::pair<double,const AdaptiveSample*>> anchors;
  for(int id:panel.samples)if(id>=0 && workspace.samples[id].reliable)
   anchors.emplace_back(workspace.samples[id].R,&workspace.samples[id]);
  std::sort(anchors.begin(),anchors.end(),[](const auto&a,const auto&b){return a.first<b.first;});
  AdaptiveSample previous;bool have_previous=false;
  for(size_t k=0;k<rule.x.size();++k){
   const double x=mid+half*rule.x[k];
   auto mapped=panel.map(x);
   if(panel.left_radius_lo!=0 || panel.right_radius_lo!=0) {
    const long double a=(long double)panel.map.a+panel.left_radius_lo;
    const long double b=(long double)panel.map.b+panel.right_radius_lo,t=(1.L+x)*.5L,w=b-a;
    constexpr long double pi=3.1415926535897932384626433832795029L;
    if(panel.map.left && panel.map.right) {
     const long double z=std::sin(pi*.5L*(t<=.5L?t:1-t));
     mapped={double(t<=.5L?a+w*z*z:b-w*z*z),double(pi*w*.25L*std::sin(pi*t))};
    } else if(panel.map.left)mapped={double(a+w*t*t),double(w*t)};
    else if(panel.map.right)mapped={double(b-w*(1-t)*(1-t)),double(w*(1-t))};
    else mapped={double(a+w*t),double(w*.5L)};
   }
   const double R=mapped[0],jac=mapped[1]*half;
   auto near=std::lower_bound(anchors.begin(),anchors.end(),R,[](const auto&a,double r){return a.first<r;});
   const AdaptiveSample*anchor=nullptr;
   auto consider=[&](const AdaptiveSample*s){if(s && (!anchor || std::fabs(s->R-R)<std::fabs(anchor->R-R)))anchor=s;};
   if(near!=anchors.end())consider(near->second);
   if(near!=anchors.begin())consider(std::prev(near)->second);
   if(have_previous)consider(&previous);
   if(primal_nodes+result.stats.node_evaluations>=cfg.max_node_evals) {
    valid=false;failure=AdaptiveStop::BudgetExceeded;break;
   }
   auto sample=eval(R,jac,panel.evaluator_cell,anchor,false,0,level,nullptr);
   ++result.stats.node_evaluations;
   if(!sample.reliable && anchor && primal_nodes+result.stats.node_evaluations<cfg.max_node_evals){sample=eval(R,jac,panel.evaluator_cell,nullptr,true,0,level,nullptr);++result.stats.node_evaluations;}
   if(!sample.reliable){valid=false;break;}
   for(int j=0;j<5;++j){valid=valid && std::isfinite(sample.value[j+1]);sum[j].add(rule.w[k]*sample.value[j+1]);}
   sample.R=R;previous=std::move(sample);have_previous=true;
  }
  if(!valid)break;
 }
 for(int j=0;j<5;++j){result.grad_mu[j]=valid?sum[j].get():std::numeric_limits<double>::quiet_NaN();result.grad_quality[j]=valid?GradientQuality::FiniteUncertified:GradientQuality::Invalid;result.grad_reason[j]=valid || failure==AdaptiveStop::BudgetExceeded?GradientReason::BudgetExceeded:GradientReason::TopologyUnresolved;}
 result.stop=valid?AdaptiveStop::Converged:failure;
 return result;
}
}
