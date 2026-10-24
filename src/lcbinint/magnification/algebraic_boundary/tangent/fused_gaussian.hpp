#pragma once
#include "geometry.hpp"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>
namespace lcbinint::algebraic_boundary::tangent::fused_gradient {
struct GaussianReplayFailure {
 AdaptiveStop reason;
 std::size_t node_evaluations;
 double radius=0;
 int rejection=-1;
 int order=0;
};
struct GaussianRule {std::vector<double> x,w;};
struct IncompleteTangentCandidate:std::runtime_error {
 std::size_t node_evaluations;
 IncompleteTangentCandidate(const char* message,std::size_t nodes):
     std::runtime_error(message),node_evaluations(nodes){}
};
inline const geometry_experiment::Pair* gaussian_panel_pair(const AdaptivePanel& panel) {
 const geometry_experiment::Pair* hot=nullptr;
 if(!geometry_experiment::active)return hot;
 const double mid=com_tangent_replay?.5*(panel.map.a+panel.map.b):
     panel.map(.5*(panel.xl+panel.xr))[0];
 for(const auto& pair:*geometry_experiment::active) {
  const double matching=std::min(pair.scale*.25,1e-10*(1+std::fabs(pair.center)));
  const bool touches_left=(com_tangent_replay||panel.xl==-1) &&
      (std::fabs(panel.map.a-pair.left)<matching||std::fabs(panel.map.a-pair.right)<matching);
  const bool touches_right=(com_tangent_replay||panel.xr==1) &&
      (std::fabs(panel.map.b-pair.left)<matching||std::fabs(panel.map.b-pair.right)<matching);
  if(pair.real && !touches_left && !touches_right)continue;
  if(mid>pair.band_lo&&mid<pair.band_hi &&
     (!hot||std::fabs(mid-pair.center)<std::fabs(mid-hot->center)))hot=&pair;
 }
 return hot;
}
inline bool gaussian_panel_split_needed(const AdaptivePanel& panel) {
 if(com_tangent_replay||!geometry_experiment::active)return false;
 auto left=panel,right=panel;
 left.xr=right.xl=.5*(panel.xl+panel.xr);
 const auto* first=gaussian_panel_pair(left);
 const auto* last=gaussian_panel_pair(right);
 return first && last && first!=last && first->real && last->real;
}
inline std::vector<AdaptivePanel> gaussian_panel_pieces(const AdaptivePanel& panel) {
 if(gaussian_panel_split_needed(panel)) {
  auto left=panel,right=panel;
  left.xr=right.xl=.5*(panel.xl+panel.xr);
  return {left,right};
 }
 return {panel};
}
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
 case 64:return fixed_gaussian_rule<64>();
 case 128:return fixed_gaussian_rule<128>();
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
inline AdaptiveResult gaussian_replay_fixed(AdaptiveWorkspace&workspace,const AdaptiveConfig&cfg,Eval&&eval,std::size_t primal_nodes,int reliability_retry=0,int forced_order=0,const std::vector<AdaptivePanel>* selected=nullptr){
 AdaptiveResult result;std::array<adaptive_detail::Sum,5> sum;
 adaptive_detail::Sum value_sum;
 bool valid=true;AdaptiveStop failure=AdaptiveStop::TopologyUnresolved;
 double failed_radius=0;
 int failed_rejection=-1;
 for(const auto&panel:selected?*selected:workspace.panels)if(panel.active){
  const auto* hot=gaussian_panel_pair(panel);
  // Mixed product integration: exact for A(x)+sqrt(1-x^2)*B(x),
  // with quadratic A and B, including the smooth background. The seven
  // fixed cosine nodes stay away from rounded physical endpoints.
  const bool pair_interior=hot&&hot->real&&!hot->upper_edge&&!hot->lower_edge&&panel.xl==-1&&panel.xr==1&&
      std::fabs(panel.map.a-hot->left)<std::min(hot->scale*.25,1e-10*(1+hot->left))&&
      std::fabs(panel.map.b-hot->right)<std::min(hot->scale*.25,1e-10*(1+hot->right));
  if(!forced_order && pair_interior){
   const int weighted_level=cfg.initial_level;
   const std::array<double,7> mixed_x{{0.92387953251128674,0.70710678118654757,0.38268343236508978,0,-0.38268343236508978,-0.70710678118654757,-0.92387953251128674}};
   const std::array<double,7> mixed_w{{0.22294318587385589,0.079567012897208009,0.7050754885397208,-0.015171374621569443,0.7050754885397208,0.079567012897208009,0.22294318587385589}};
   AdaptiveSample previous;bool have_previous=false;
   const long double a=(long double)panel.map.a+panel.left_radius_lo;
   const long double b=(long double)panel.map.b+panel.right_radius_lo;
   for(size_t k=0;k<mixed_x.size();++k){
    const double x=mixed_x[k];
    const long double radius=a+(b-a)*(1.L+x)*.5L;
    const double R=double(radius),lo=double(radius-(long double)R),jac=double((b-a)*.5L);
    if(primal_nodes+result.stats.node_evaluations>=cfg.max_node_evals){valid=false;failure=AdaptiveStop::BudgetExceeded;break;}
    auto sample=eval(R,jac,panel.evaluator_cell,have_previous?&previous:nullptr,false,lo,weighted_level,nullptr);
    ++result.stats.node_evaluations;
    if(!sample.reliable&&have_previous&&primal_nodes+result.stats.node_evaluations<cfg.max_node_evals){sample=eval(R,jac,panel.evaluator_cell,nullptr,true,lo,weighted_level,nullptr);++result.stats.node_evaluations;}
    if(!sample.reliable){valid=false;break;}
    value_sum.add(mixed_w[k]*sample.value[0]);
    for(int j=0;j<5;++j){
     const double term=mixed_w[k]*sample.value[j+1];
     valid=valid&&std::isfinite(term);sum[j].add(term);
    }
    sample.R=R;previous=std::move(sample);have_previous=true;
   }
   if(!valid)break;
   continue;
  }
  if(!forced_order && panel.depth==0&&!hot&&!(geometry_experiment::active&&!geometry_experiment::active->empty())){value_sum.add(panel.q[0]);for(int j=0;j<5;++j)sum[j].add(panel.q[j+1]);continue;}

  const int level=std::max(cfg.initial_level,panel.level);
  const auto&rule=gaussian_rule(forced_order?forced_order:(reliability_retry?14:(hot?64:((1<<level)-1)*2)));
  const double half=.5*(panel.xr-panel.xl),mid=.5*(panel.xr+panel.xl);
  std::vector<std::pair<double,const AdaptiveSample*>> anchors;
  for(int id:panel.samples)if(id>=0 && workspace.samples[id].reliable)
   anchors.emplace_back(workspace.samples[id].R,&workspace.samples[id]);
  std::sort(anchors.begin(),anchors.end(),[](const auto&a,const auto&b){return a.first<b.first;});
  AdaptiveSample previous;bool have_previous=false;
  for(size_t k=0;k<rule.x.size();++k){
   const double x=mid+half*rule.x[k];
   auto map=panel.map;if(hot&&panel.weighted)map.left=map.right=true;
   if(panel.depth==0||!cfg.fold_maps)map.left=map.right=true;
   auto mapped=map(x);
   if(panel.left_radius_lo!=0 || panel.right_radius_lo!=0) {
    const long double a=(long double)panel.map.a+panel.left_radius_lo;
    const long double b=(long double)panel.map.b+panel.right_radius_lo,t=(1.L+x)*.5L,w=b-a;
    constexpr long double pi=3.1415926535897932384626433832795029L;
    if(map.left && map.right) {
     const long double z=std::sin(pi*.5L*(t<=.5L?t:1-t));
     mapped={double(t<=.5L?a+w*z*z:b-w*z*z),double(pi*w*.25L*std::sin(pi*t))};
    } else if(map.left)mapped={double(a+w*t*t),double(w*t)};
    else if(map.right)mapped={double(b-w*(1-t)*(1-t)),double(w*(1-t))};
    else mapped={double(a+w*t),double(w*.5L)};
   }
   double jac=mapped[1]*half;
   if(hot&&!hot->real){
    const double lo=map(panel.xl)[0],hi=map(panel.xr)[0];
    FoldRadialMap hyperbolic{std::asinh((lo-hot->center)/hot->scale),std::asinh((hi-hot->center)/hot->scale),map.left&&panel.xl==-1,map.right&&panel.xr==1};
    const auto y=hyperbolic(rule.x[k]);mapped[0]=hot->center+hot->scale*std::sinh(y[0]);jac=hot->scale*std::cosh(y[0])*y[1];
   }
   double radius_lo=0;
   if(!hot) {
    const long double a=(long double)panel.map.a+panel.left_radius_lo;
    const long double b=(long double)panel.map.b+panel.right_radius_lo;
    const long double t=(1.L+x)*.5L,w=b-a;
    long double radius;
    if(map.left&&map.right) {
     const long double z=sinl(kAngularPi*.5L*(t<=.5L?t:1-t));
     radius=t<=.5L?a+w*z*z:b-w*z*z;
    } else if(map.left)radius=a+w*t*t;
    else if(map.right)radius=b-w*(1-t)*(1-t);
    else radius=a+w*t;
    mapped[0]=double(radius);
    radius_lo=double(radius-(long double)mapped[0]);
   }
   if(hot&&hot->real){
    const double matching=std::min(hot->scale*.25,1e-10*(1+std::fabs(hot->center)));
    const bool left=std::fabs(panel.map.b-hot->left)<matching;
    const bool right=std::fabs(panel.map.a-hot->right)<matching;
    if(left||right){
     const long double a=(long double)panel.map.a+panel.left_radius_lo;
     const long double b=(long double)panel.map.b+panel.right_radius_lo;
     auto bound=[&](double x){if(x==-1)return a;if(x==1)return b;const long double t=(1.L+x)*.5L,w=b-a;
      if(map.left&&map.right){const long double s=sinl(3.1415926535897932384626433832795029L*t*.5L);return a+w*s*s;}
      if(map.left)return a+w*t*t;if(map.right)return b-w*(1-t)*(1-t);return a+w*t;
     };
     const long double lo=bound(panel.xl),hi=bound(panel.xr),critical=left?b:a,scale=hot->scale;
     auto y=[&](long double R){const long double distance=left?critical-R:R-critical;return 2*asinhl(sqrtl(distance/(2*scale)));};
     FoldRadialMap hyperbolic{double(y(lo)),double(y(hi)),
        map.left&&panel.xl==-1&&!right,map.right&&panel.xr==1&&!left};
     auto yy=hyperbolic(rule.x[k]);const long double sh=sinhl((long double)yy[0]*.5L);
     const long double radius=critical+(left?-1:1)*2*scale*sh*sh;
     mapped[0]=double(radius);radius_lo=double(radius-(long double)mapped[0]);
     jac=double((left?-1:1)*scale*sinhl((long double)yy[0])*yy[1]);
    }
   }
   const double R=mapped[0];
   auto near=std::lower_bound(anchors.begin(),anchors.end(),R,[](const auto&a,double r){return a.first<r;});
   const AdaptiveSample*anchor=nullptr;
   auto consider=[&](const AdaptiveSample*s){if(s && (!anchor || std::fabs(s->R-R)<std::fabs(anchor->R-R)))anchor=s;};
   if(near!=anchors.end())consider(near->second);
   if(near!=anchors.begin())consider(std::prev(near)->second);
   if(have_previous)consider(&previous);
   if(primal_nodes+result.stats.node_evaluations>=cfg.max_node_evals) {
    valid=false;failure=AdaptiveStop::BudgetExceeded;break;
   }
   auto sample=eval(R,jac,panel.evaluator_cell,anchor,false,radius_lo,level,nullptr);
   ++result.stats.node_evaluations;
   if(!sample.reliable && anchor && primal_nodes+result.stats.node_evaluations<cfg.max_node_evals){sample=eval(R,jac,panel.evaluator_cell,nullptr,true,radius_lo,level,nullptr);++result.stats.node_evaluations;}
   if(!sample.reliable){valid=false;failed_radius=R;failed_rejection=int(sample.reject_reason);break;}
   value_sum.add(rule.w[k]*sample.value[0]);
   for(int j=0;j<5;++j){valid=valid && std::isfinite(sample.value[j+1]);sum[j].add(rule.w[k]*sample.value[j+1]);}
   sample.R=R;previous=std::move(sample);have_previous=true;
  }
  if(!valid)break;
 }
 // A desingularized panel may place its highest-order nodes closer to a
 // fold than its endpoint roots can resolve. Retry a fixed interior rule only
 // after geometry evaluation failed; this does not inspect gradient errors.
 if(!forced_order&&!valid&&failure==AdaptiveStop::TopologyUnresolved&&reliability_retry==0){
  reliability_retry_used=true;
  try {
   auto retry=gaussian_replay_fixed(workspace,cfg,eval,primal_nodes+result.stats.node_evaluations,1);
   retry.stats.node_evaluations+=result.stats.node_evaluations;return retry;
  } catch(GaussianReplayFailure& retry) {
   retry.node_evaluations+=result.stats.node_evaluations;throw;
  }
 }
 for(int j=0;j<5;++j)valid=valid&&std::isfinite(sum[j].get());
 if(!valid)throw GaussianReplayFailure{failure,result.stats.node_evaluations,failed_radius,failed_rejection,forced_order};
 for(int j=0;j<5;++j){result.grad_mu[j]=sum[j].get();result.grad_quality[j]=GradientQuality::FiniteUncertified;result.grad_reason[j]=GradientReason::BudgetExceeded;}
 result.mu=value_sum.get();
 result.stop=AdaptiveStop::Converged;
 return result;
}

template<class Eval>
inline AdaptiveResult gaussian_replay(AdaptiveWorkspace& workspace,const AdaptiveConfig& cfg,
    Eval&& eval,std::size_t primal_nodes) {
 const PreciseTangentScope precision_scope;
 if(!cfg.gradient_adaptive) {
  AdaptiveResult result;
  AdaptiveConfig replay_config=cfg;
  const auto active_panels=std::count_if(workspace.panels.begin(),workspace.panels.end(),
      [](const auto& panel){return panel.active;});
  if(active_panels==0)throw GaussianReplayFailure{AdaptiveStop::TopologyUnresolved,0};
  std::size_t completion_nodes=0;
  for(const auto& panel:workspace.panels)if(panel.active)
   completion_nodes+=216*(gaussian_panel_split_needed(panel)?2:1);
  replay_config.max_node_evals=std::max(cfg.max_node_evals,primal_nodes+completion_nodes);
  std::array<adaptive_detail::Sum,5> sums;
  adaptive_detail::Sum value_sum;
  for(const auto& panel:workspace.panels)if(panel.active) {
   const auto selected=gaussian_panel_pieces(panel);
   bool computed=false;
   GaussianReplayFailure last_failure{AdaptiveStop::TopologyUnresolved,0};
   std::array<double,5> candidate{};
   double candidate_value=0;
   for(int order:{64,30,14}) {
    try {
     const auto piece=gaussian_replay_fixed(workspace,replay_config,eval,
         primal_nodes+result.stats.node_evaluations,0,order,&selected);
     result.stats.node_evaluations+=piece.stats.node_evaluations;
     candidate=piece.grad_mu;candidate_value=piece.mu;computed=true;break;
    } catch(const GaussianReplayFailure& failure) {
     result.stats.node_evaluations+=failure.node_evaluations;
     last_failure=failure;
    }
   }
   if(!computed) {
    if(cfg.fused_value_mesh||panel.depth!=0||panel.level<=0) {
     last_failure.node_evaluations=result.stats.node_evaluations;
     throw last_failure;
    }
    candidate_value=panel.q[0];
    for(int axis=0;axis<5;++axis) {
     candidate[axis]=panel.q[axis+1];
     if(!std::isfinite(candidate[axis]))
      throw GaussianReplayFailure{AdaptiveStop::TopologyUnresolved,result.stats.node_evaluations};
    }
   }
   value_sum.add(candidate_value);
   for(int axis=0;axis<5;++axis)sums[axis].add(candidate[axis]);
  }
  for(int axis=0;axis<5;++axis) {
   result.grad_mu[axis]=sums[axis].get();
   result.grad_quality[axis]=GradientQuality::FiniteUncertified;
   result.grad_reason[axis]=GradientReason::BudgetExceeded;
   result.estimated_abs_error_grad[axis]=std::numeric_limits<double>::infinity();
  }
  result.mu=value_sum.get();
  result.stop=AdaptiveStop::Converged;
  return result;
 }
 struct Piece {
  AdaptivePanel panel;
  std::array<double,5> value{},error{};
  int depth=0;
  bool blocked=false;
 };
 std::size_t used=0;
 auto estimate=[&](const AdaptivePanel& panel,int depth,Piece& piece,bool incumbent) {
  piece.panel=panel;piece.depth=depth;
  std::array<double,5> low{},high{};
  bool have_low=false,have_high=false;
  const std::vector<AdaptivePanel> selected{panel};
  for(int order:{14,30}) {
   try {
    auto candidate=gaussian_replay_fixed(workspace,cfg,eval,primal_nodes+used,0,order,&selected);
    used+=candidate.stats.node_evaluations;
    if(order==14){low=candidate.grad_mu;have_low=true;}
    else {high=candidate.grad_mu;have_high=true;}
   } catch(const GaussianReplayFailure& failure) {used+=failure.node_evaluations;}
  }
  if(!have_high&&!have_low) {
   if(!incumbent||panel.depth>0||panel.level<=0)return false;
   for(int axis=0;axis<5;++axis) {
    high[axis]=panel.q[axis+1];
    if(!std::isfinite(high[axis]))return false;
   }
  } else if(!have_high)high=low;
  piece.value=high;
  for(int axis=0;axis<5;++axis)
   piece.error[axis]=have_high&&have_low?2*std::fabs(high[axis]-low[axis])
       :std::numeric_limits<double>::infinity();
  return true;
 };
 std::vector<Piece> pieces;
 for(const auto& panel:workspace.panels)if(panel.active) {
  Piece piece;
  if(!estimate(panel,0,piece,true))throw GaussianReplayFailure{AdaptiveStop::TopologyUnresolved,used};
  pieces.push_back(std::move(piece));
 }
 if(pieces.empty())throw GaussianReplayFailure{AdaptiveStop::TopologyUnresolved,used};
 AdaptiveResult result;
 bool converged=false;
 for(;;) {
  std::array<adaptive_detail::Sum,5> sums;
  std::array<double,5> errors{},refinable_errors{};
  for(const auto& piece:pieces)for(int axis=0;axis<5;++axis) {
   sums[axis].add(piece.value[axis]);errors[axis]+=piece.error[axis];
   if(!piece.blocked&&piece.depth<cfg.max_depth)refinable_errors[axis]+=piece.error[axis];
  }
  converged=true;
  std::array<double,5> targets{};
  for(int axis=0;axis<5;++axis) {
   result.grad_mu[axis]=sums[axis].get();
   targets[axis]=cfg.tol.budget(axis+1,result.grad_mu[axis]);
   converged=converged&&errors[axis]<=targets[axis];
   result.estimated_abs_error_grad[axis]=errors[axis];
  }
  bool refinement_needed=false;
  for(int axis=0;axis<5;++axis)refinement_needed=refinement_needed||refinable_errors[axis]>targets[axis];
  if(converged||!refinement_needed||primal_nodes+used+88>cfg.max_node_evals||pieces.size()+1>cfg.max_panels)break;
  std::size_t selected=pieces.size();double worst=0;
  for(std::size_t index=0;index<pieces.size();++index) {
   const auto& piece=pieces[index];
   if(piece.blocked||piece.depth>=cfg.max_depth)continue;
   double score=0;
   for(int axis=0;axis<5;++axis) {
    const double scaled=targets[axis]>0?piece.error[axis]/targets[axis]
        :(piece.error[axis]>0?std::numeric_limits<double>::infinity():0.0);
    score=std::max(score,scaled);
   }
   if(score>worst){worst=score;selected=index;}
  }
  if(selected==pieces.size())break;
  const auto parent=pieces[selected];
  auto left=parent.panel,right=parent.panel;
  const double midpoint=.5*(left.xl+right.xr);
  if(!(midpoint>left.xl&&midpoint<right.xr)){pieces[selected].blocked=true;continue;}
  left.xr=midpoint;right.xl=midpoint;
  Piece first,second;
  const bool have_first=estimate(left,parent.depth+1,first,false);
  const bool have_second=estimate(right,parent.depth+1,second,false);
  if(!have_first||!have_second){pieces[selected].blocked=true;continue;}
  pieces[selected]=std::move(first);pieces.push_back(std::move(second));
 }
 result.stats.node_evaluations=used;
 result.stop=converged?AdaptiveStop::Converged:AdaptiveStop::BudgetExceeded;
 for(int axis=0;axis<5;++axis) {
  result.grad_quality[axis]=converged?GradientQuality::ToleranceMet:GradientQuality::FiniteUncertified;
  result.grad_reason[axis]=converged?GradientReason::ToleranceMet:GradientReason::BudgetExceeded;
 }
 return result;
}
}
