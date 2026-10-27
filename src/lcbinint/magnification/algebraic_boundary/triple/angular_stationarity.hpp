#pragma once
#include "point_polynomial.hpp"
#include "local_precision.hpp"
#include <optional>
#include <vector>
namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail {
// phi=P6/(rho^2 R^2 D6). S10=P6'D6-P6D6' has the same stationary
// roots; its degree-eleven term cancels symbolically. Coefficient balls and
// Bernstein variations certify candidates. Ambiguity retains the incumbent
// physical search. The whole circle is isolated once, shared by its arcs.
class AngularStationarity {
 using B=coefficient_ball_detail::Ball<double>;using Poly=std::array<B,11>;
 const Geometry& p_;double radius_;Poly coefficients_{};
 mutable bool attempted_=false;mutable std::optional<std::vector<std::array<double,2>>> roots_;
 static int choose(int n,int k){int out=1;for(int j=1;j<=k;++j)out=out*(n-j+1)/j;return out;}
 static int sign(B x){if(!std::isfinite(x.value)||!std::isfinite(x.error)||x.error<0)return 2;return x.value>x.error?1:x.value<-x.error?-1:x.zero()?0:2;}
 static int variations(const Poly& b){int previous=0,count=0;for(auto x:b){const int s=sign(x);if(s==2)return -1;if(!s)continue;if(previous&&s!=previous)++count;previous=s;}return count;}
 struct Weight {double value,error;};
 static const auto& matrix(){static const auto m=[](){std::array<std::array<Weight,11>,11> out{};for(int i=0;i<=10;++i)for(int j=0;j<=10;++j){long double sum=0,magnitude=0;for(int k=0;k<=i&&k<=j;++k){const long double term=(long double)choose(j,k)*std::ldexp(1.L,k)*choose(i,k)/choose(10,k);sum+=(j-k)%2?-term:term;magnitude+=term;}const double v=double(sum);out[i][j]={v,double(std::abs(sum-(long double)v)+1024*std::numeric_limits<long double>::epsilon()*magnitude)};}return out;}();return m;}
 static Poly restrict_full(const Poly& p){Poly out{};const auto& m=matrix();const double eps=std::numeric_limits<double>::epsilon();for(int i=0;i<=10;++i){double value=0,error=0,magnitude=0;for(int j=0;j<=10;++j){const double product=m[i][j].value*p[j].value;value+=product;magnitude+=std::abs(product);error+=std::abs(m[i][j].value)*p[j].error+m[i][j].error*(std::abs(p[j].value)+p[j].error);}out[i]={value,(error+64*eps*magnitude+64*std::numeric_limits<double>::min())*(1+256*eps)};}return out;}
 static std::pair<Poly,Poly> split(Poly b){Poly left{},right{};left[0]=b[0];right[10]=b[10];const double eps=std::numeric_limits<double>::epsilon();for(int level=1;level<=10;++level){for(int j=0;j<=10-level;++j){const double v=.5*(b[j].value+b[j+1].value),e=.5*(b[j].error+b[j+1].error+eps*(std::abs(b[j].value)+std::abs(b[j+1].value)));b[j]={v,(e+std::numeric_limits<double>::min())*(1+32*eps)};}left[level]=b[0];right[10-level]=b[10-level];}return {left,right};}
 std::optional<std::vector<std::array<double,2>>> brackets()const{
  const double pi=std::acos(-1.);std::vector<std::array<double,2>> result;int boxes=0;
  for(int chart=0;chart<2;++chart){Poly p=coefficients_;if(chart)for(int j=0;j<=10;++j)p[j]=j%2?-coefficients_[10-j]:coefficients_[10-j];
   struct Box {double lo,hi;Poly b;};std::array<Box,33> stack{};int size=1;stack[0]={-1,1,restrict_full(p)};
   while(size){if(++boxes>32)return std::nullopt;auto box=std::move(stack[--size]);const int n=variations(box.b);if(n==0)continue;
    if(n==1){const int l=sign(box.b.front()),r=sign(box.b.back());if(l==1&&r==-1)continue;if(l==-1&&r==1){result.push_back({chart*pi+2*std::atan(box.lo),chart*pi+2*std::atan(box.hi)});continue;}}
    const double mid=box.lo+(box.hi-box.lo)*.5;if(!(mid>box.lo&&mid<box.hi)||size+2>int(stack.size()))return std::nullopt;auto halves=split(box.b);stack[size++]={mid,box.hi,std::move(halves.second)};stack[size++]={box.lo,mid,std::move(halves.first)};
   }
  }return result;
 }
public:
 AngularStationarity(double R,const Geometry& p,const point_polynomial_detail::Cache<double>& cache):p_(p),radius_(R){
  const B r(R),a(p.a),bx(p.bx),by(p.by);const auto pc=cache.at(R).c;
  const std::array<std::array<B,3>,3> factors{{{B(1),B(0),B(1)},{(r-a)*(r-a),B(0),(r+a)*(r+a)},{(r-bx)*(r-bx)+by*by,B(-4)*r*by,(r+bx)*(r+bx)+by*by}}};
  std::array<B,7> denominator{};denominator[0]=B(1);int degree=0;for(const auto& factor:factors){std::array<B,7> next{};for(int i=0;i<=degree;++i)for(int j=0;j<3;++j)next[i+j]=next[i+j]+denominator[i]*factor[j];denominator=next;degree+=2;}
  for(int i=0;i<=6;++i)for(int j=0;j<=6;++j)if(i!=j)coefficients_[i+j-1]=coefficients_[i+j-1]+B(i-j)*pc[i]*denominator[j];
 }
 template<class Ray,class Work> std::optional<std::vector<std::array<double,4>>> minima(const Ray& ray,Work& work,double lo,double hi)const{
  if(!attempted_){attempted_=true;roots_=brackets();}if(!roots_)return std::nullopt;std::vector<std::array<double,4>> result;
  const double pi=std::acos(-1.);const bool full=hi-lo>=2*pi-1e-12;
  for(auto bracket:*roots_){const double shift=2*pi*std::round(((lo+hi)*.5-(bracket[0]+bracket[1])*.5)/(2*pi));bracket[0]+=shift;bracket[1]+=shift;
   // A physical positive arc enters with phi_theta>0 and leaves with
   // phi_theta<0. A unique (-,+) stationary bracket crossing either end
   // therefore has its minimum outside this arc. No physical Newton work
   // is needed for the many negative minima elsewhere on the same circle.
   if(!full&&(bracket[0]<lo||bracket[1]>hi))continue;
   double a=bracket[0],b=bracket[1],c=.5*(a+b),curvature=0,error=0;Evaluation<true> jet{};
   for(int iteration=0;iteration<24;++iteration){jet=ray.template sample<false>(c,error,&curvature);if(jet.dtheta<0)a=c;else b=c;const double next=c-jet.dtheta/curvature;if(std::abs(next-c)<1e-13*(1+std::abs(c)))break;c=std::isfinite(next)&&next>a&&next<b?next:.5*(a+b);}
   jet=ray.template sample<false>(c,error,&curvature);if(!(curvature>0)||!std::isfinite(jet.phi)||!std::isfinite(jet.dtheta)||std::abs(jet.dtheta)>1e-10*std::max(1.,std::abs(curvature)))return std::nullopt;
   double phi=jet.phi;if(phi<-2*error)return std::nullopt;
   if(!(phi>0)||error>1e-5*phi){const auto q=local_precision::evaluate<false>(local_precision::Q(radius_),local_precision::Q(c),p_);phi=double(q.phi);error=64*std::numeric_limits<double>::epsilon()*std::abs(phi);++work.quad_samples;}
   if(phi>0){const double scale=std::sqrt(2*phi/curvature);if(std::isfinite(scale))result.push_back({c,scale,phi,error});}
  }return result;
 }
};
}
