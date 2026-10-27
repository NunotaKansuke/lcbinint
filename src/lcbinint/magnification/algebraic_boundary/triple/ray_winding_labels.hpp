#pragma once
#include "point_polynomial.hpp"
#include "physical_jet.hpp"
#include "local_precision.hpp"
namespace lcbinint::algebraic_boundary::triple::winding_label_detail {
using B=coefficient_ball_detail::Ball<double>;
inline constexpr auto binomials=[](){std::array<std::array<int,9>,9> b{};for(int n=0;n<=8;++n){b[n][0]=b[n][n]=1;for(int k=1;k<n;++k)b[n][k]=b[n-1][k-1]+b[n-1][k];}return b;}();
class Plan {
 Geometry p_;double coordinate_=0;bool reciprocal_=false;std::array<B,9> polynomial_{},angular_{};int nodes_=0;
 struct Tile{double lo,hi;int tag=0,offset=0,region=0;};std::vector<Tile> tiles_;
 static int sign(const std::array<B,9>& p){bool pos=true,neg=true;for(const auto& x:p){const double pad=16*std::numeric_limits<double>::epsilon()*(std::abs(x.value)+x.error)+16*std::numeric_limits<double>::min();pos&=std::isfinite(x.value)&&std::isfinite(x.error)&&x.value-x.error-pad>0;neg&=std::isfinite(x.value)&&std::isfinite(x.error)&&x.value+x.error+pad<0;}return pos?1:neg?-1:0;}
 static std::array<B,9> bernstein(const std::array<B,9>& p,double lo,double hi){const B a(lo),w=B(hi)-B(lo);std::array<B,9> ap{},wp{},q{},b{};ap[0]=wp[0]=B(1);for(int j=1;j<=8;++j){ap[j]=ap[j-1]*a;wp[j]=wp[j-1]*w;}for(int j=0;j<=8;++j)for(int k=j;k<=8;++k)q[j]=q[j]+p[k]*B(binomials[k][j])*ap[k-j]*wp[j];for(int i=0;i<=8;++i)for(int j=0;j<=i;++j){const double r=double(binomials[i][j])/binomials[8][j];b[i]=b[i]+q[j]*B(r,4*std::numeric_limits<double>::epsilon()*r);}return b;}
 static void halves(std::array<B,9> p,std::array<B,9>& left,std::array<B,9>& right){left[0]=p[0];right[8]=p[8];for(int k=1;k<=8;++k){for(int j=0;j<=8-k;++j)p[j]=(p[j]+p[j+1])*B(.5);left[k]=p[0];right[8-k]=p[8-k];}}
 void partition(double lo,double hi,const std::array<B,9>& a,const std::array<B,9>& b,int depth){++nodes_;const int sa=sign(a),sb=sign(b),tag=sa?sa:sb?2*sb:0;const double mid=lo+(hi-lo)*.5;if(tag||depth==18||nodes_>=512||mid==lo||mid==hi){tiles_.push_back({lo,hi,tag});return;}std::array<B,9> al{},ar{},bl{},br{};halves(a,al,ar);halves(b,bl,br);partition(lo,mid,al,bl,depth+1);partition(mid,hi,ar,br,depth+1);}
 const Tile* tile(double R)const{const auto i=std::lower_bound(tiles_.begin(),tiles_.end(),R,[](const Tile& t,double r){return t.hi<r;});return i!=tiles_.end()&&R>=i->lo?&*i:nullptr;}
 int physical_sign(double R,std::size_t* quad)const{
  const auto f=point_polynomial_detail::horner_ball<double>(8,R,[&](int j){return polynomial_[j];});if(f.value>f.error)return 1;if(f.value<-f.error)return -1;
  using Q=local_precision::Q;const auto value=physical_detail::Ray<Q>(Q(R),p_).value(Q(coordinate_),reciprocal_);if(quad)++*quad;if(!value||!finiteq(*value))return 2;return *value>0?1:*value<0?-1:0;
 }
public:
 double angle=0;Complex unit{};std::size_t tile_count()const{return tiles_.size();}
 Plan(const Geometry& p,const point_polynomial_detail::Cache<double>& tape,double lo,double hi):p_(p){
  if(!(hi>lo&&lo>0))return;angle=.123;for(double trial:{.123,.731,1.337,2.113}){const Complex u=std::polar(1.,trial);bool clear=true;for(const auto& lens:p.lenses())if(std::norm(lens)>0&&std::real(std::conj(u)*lens)>0&&std::abs(std::imag(std::conj(u)*lens))<1e-6*std::abs(lens))clear=false;if(clear){angle=trial;break;}}
  coordinate_=std::tan(angle*.5);reciprocal_=std::abs(coordinate_)>1;if(reciprocal_)coordinate_=-1/coordinate_;const double s=reciprocal_?-1.:1.;unit=s*Complex(1-coordinate_*coordinate_,2*coordinate_);angle=std::arg(unit);
  std::array<B,7> powers{};powers[0]=B(1);for(int k=1;k<=6;++k)powers[k]=powers[k-1]*B(coordinate_);
  for(int k=0;k<7;++k){const auto& c=tape.coefficients[reciprocal_?6-k:k];const B sgn(reciprocal_&&k%2?-1.:1.);for(int j=0;j<=c.degree;++j){polynomial_[j]=polynomial_[j]+c.c[j]*powers[k]*sgn;if(k)angular_[j]=angular_[j]+c.c[j]*B(k)*powers[k-1]*sgn;}}
  partition(lo,hi,bernstein(polynomial_,lo,hi),bernstein(angular_,lo,hi),0);
  int region=0,offset=0,previous=0;for(auto& t:tiles_){if(!t.tag){t.region=-1;++region;offset=0;previous=0;continue;}if(previous==-1&&t.tag==-2)++offset;if(previous==-2&&t.tag==-1)--offset;t.offset=offset;t.region=region;previous=t.tag;}
 }
 bool covers(double lo,double hi)const{const auto a=tile(lo),b=tile(hi);if(!a||!b||!a->tag||!b->tag||a->region!=b->region)return false;return true;}
 // Lift arg(P+i P_t) through certified half-planes. At a real root P_t
 // has the physical angular orientation; crossings of +/-pi/2 therefore
 // count lower/upper image endpoints. No polynomial root isolation or
 // per-image completeness proof is repeated.
 std::optional<std::array<int,3>> indices(double R,std::size_t* quad=nullptr)const{
  const auto t=tile(R);if(!t||!t->tag)return std::nullopt;const int s=t->tag==1?1:t->tag==-1?-1:physical_sign(R,quad);if(s==2)return std::nullopt;int lower=0,upper=0;
  if(t->tag==1){lower=-1;upper=0;}else if(t->tag==-1){lower=upper=0;}else if(t->tag==2){lower=s<=0?0:-1;upper=0;}else{lower=-1;upper=s<0?-1:0;}return std::array<int,3>{lower+t->offset,upper+t->offset,s};
 }
};
}
