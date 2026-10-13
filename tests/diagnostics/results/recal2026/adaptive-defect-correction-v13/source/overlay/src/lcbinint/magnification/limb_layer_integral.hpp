#pragma once
#include <array>
#include <cmath>
#include <map>
#include <vector>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <limits>
namespace lcbinint::magnification::detail {
// Exact layer-cake identity for B=1-c-d+c*mu+d*sqrt(mu), mu=t^2:
// flux=(1-c-d) A(rho)+integral_0^1 (2ct+d) A(rho sqrt(1-t^4)) dt.
// Also t=1-u^2 regularizes A(r)~r at a centred single-lens caustic.
// Disk-area provider owns geometric state. This class never discovers images.
class LimbLayerIntegral {
public:
 using Provider=std::function<std::array<double,2>(double,double)>;
 struct Node {double t,value,error;};
 struct Leaf {double lo,hi,area,error;unsigned depth;};
 LimbLayerIntegral(double radius,double c,double d,Provider provider):rho_(radius),c_(c),d_(d),provider_(std::move(provider)) {
  if(!std::isfinite(radius)||radius<=0||!std::isfinite(c)||!std::isfinite(d)||!(1-c/3-d/5>0))throw std::invalid_argument("invalid LD parameters");
 }
 bool refine(double tolerance,std::size_t maximum_calls) {
  if(!(tolerance>0)||!std::isfinite(tolerance))throw std::invalid_argument("invalid flux tolerance");
  if(!initialized_) {
   if(maximum_calls<(c_==0&&d_==0?1u:16u))return false;
   area_budget_=tolerance/(16*(1+std::abs(c_)+std::abs(d_)));
   const auto outer=provider_(rho_,area_budget_);++calls;
   if(!valid(outer))return false;
   outer_=outer;outer_flux_=(1-c_-d_)*outer[0];outer_error_=std::abs(1-c_-d_)*outer[1];
   if(c_==0&&d_==0){initialized_=true;return error()<=tolerance;}
   // u=0: zero radius exactly. u=1: reuse the outer disk.
   nodes_[0]={0,0,0};nodes_[1]={1,2*d_*outer[0],2*std::abs(d_)*outer[1]};
   auto leaf=make(0,1,0);if(!std::isfinite(leaf.area))return false;
   leaves_.push_back(leaf);initialized_=true;
  }
  const double requested=tolerance/(16*(1+std::abs(c_)+std::abs(d_)));
  if(requested<area_budget_) {
   // Provider may refine a retained disk state; never require a new seed/fill.
   if(outer_[1]>requested){if(calls>=maximum_calls)return false;auto v=provider_(rho_,requested);++calls;if(!valid(v))return false;outer_=v;outer_flux_=(1-c_-d_)*v[0];outer_error_=std::abs(1-c_-d_)*v[1];if(nodes_.count(1))nodes_[1]={1,2*d_*v[0],2*std::abs(d_)*v[1]};}
   bool complete=true;
   for(auto& [t,n]:nodes_)if(t>0&&t<1&&n.error>std::abs(weight(t))*requested){if(calls>=maximum_calls){complete=false;break;}const double r=radius(t);auto v=provider_(r,requested);++calls;if(!valid(v)){complete=false;break;}const double w=weight(t);n={t,w*v[0],std::abs(w)*v[1]};}
   for(auto& p:leaves_)p=make(p.lo,p.hi,p.depth);
   if(!complete)return false;area_budget_=requested;
  }
  while(error()>tolerance) {
   std::size_t best=leaves_.size();double score=-1;
   for(std::size_t i=0;i<leaves_.size();++i)if(leaves_[i].depth<20&&leaves_[i].error>score){score=leaves_[i].error;best=i;}
   if(best==leaves_.size()||calls+30>maximum_calls)return false;
   const auto parent=leaves_[best];const double mid=(parent.lo+parent.hi)/2;
   auto lo=make(parent.lo,mid,parent.depth+1),hi=make(mid,parent.hi,parent.depth+1);
   if(!std::isfinite(lo.area)||!std::isfinite(hi.area))return false;
   leaves_[best]=lo;leaves_.push_back(hi);++splits;
  }
  return std::isfinite(error());
 }
 double flux()const {if(!initialized_)return NAN;double s=outer_flux_;for(auto p:leaves_)s+=p.area;return s;}
 double error()const {if(!initialized_)return INFINITY;double s=outer_error_;for(auto p:leaves_)s+=p.error;return s;}
 double source_flux()const{return 3.141592653589793*rho_*rho_*(1-c_/3-d_/5);}
 std::size_t calls=0,splits=0;
 std::size_t retained_nodes()const{return nodes_.size();}
private:
 double rho_,c_,d_,area_budget_=0,outer_flux_=0,outer_error_=0;Provider provider_;bool initialized_=false;
 std::array<double,2> outer_{};std::map<double,Node> nodes_;std::vector<Leaf> leaves_;
 static bool valid(std::array<double,2> v){return std::isfinite(v[0])&&std::isfinite(v[1])&&v[0]>=0&&v[1]>=0;}
 Node node(double t){auto found=nodes_.find(t);if(found!=nodes_.end())return found->second;
  const double r=radius(t);auto v=provider_(r,area_budget_);++calls;
  const double w=weight(t);Node n{t,valid(v)?w*v[0]:NAN,valid(v)?std::abs(w)*v[1]:INFINITY};nodes_[t]=n;return n;
 }
 double radius(double u)const {const double t=1-u*u;return rho_*u*std::sqrt((2-u*u)*(1+t*t));}
 double weight(double u)const{return 2*u*(2*c_*(1-u*u)+d_);}
 Leaf make(double a,double b,unsigned depth){
  // Embedded Gauss-7 / Kronrod-15: one set of disk areas supplies both rules.
  constexpr double xs[8]={.991455371120812639,.949107912342758524,.864864423359769073,.741531185599394439,.586087235467691130,.405845151377397167,.207784955007898468,0};
  constexpr double kw[8]={.022935322010529225,.063092092629978553,.104790010322250184,.140653259715525919,.169004726639267903,.190350578064785410,.204432940075298892,.209482141084727828};
  constexpr double gw[4]={.129484966168869693,.279705391489276668,.381830050505118945,.417959183673469388};
  const double m=(a+b)/2,h=(b-a)/2;const auto center=node(m);
  double fine=kw[7]*center.value,coarse=gw[3]*center.value,noise=kw[7]*center.error,absolute=kw[7]*std::abs(center.value);
  for(int i=0;i<7;++i){const auto l=node(m-h*xs[i]),r=node(m+h*xs[i]);const double sum=l.value+r.value;
   fine+=kw[i]*sum;noise+=kw[i]*(l.error+r.error);absolute+=kw[i]*(std::abs(l.value)+std::abs(r.value));
   if(i%2==1)coarse+=gw[i/2]*sum;
  }
  return {a,b,h*fine,2*h*std::abs(fine-coarse)+h*noise+64*std::numeric_limits<double>::epsilon()*h*absolute,depth};
 }
};
}
