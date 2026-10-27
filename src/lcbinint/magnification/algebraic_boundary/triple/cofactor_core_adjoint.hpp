#pragma once
#include "pair_moments.hpp"
namespace lcbinint::algebraic_boundary::triple::root_pair_detail {
// Scalar analytic cofactor series, followed by a single canonical pullback.
// The four polynomial logarithmic derivatives replace eight AD recurrences.
template<int N> class CofactorCoreAdjoint {
 const Arc<N>* model_=nullptr;
 std::array<std::array<double,5>,4> shifted_{};
 std::array<std::array<double,17>,4> divided_{};
public:
 CofactorSeries<0> series;bool valid=false;
 explicit CofactorCoreAdjoint(const Arc<N>& model,double gate=1e-8):model_(&model){
  Arc<0> scalar;scalar.pair.mean.value=model.pair.mean.value;scalar.pair.square.value=model.pair.square.value;scalar.half_width=model.half_width;scalar.rho.value=model.rho.value;scalar.inverse_rho=model.inverse_rho;
  for(int j=0;j<5;++j)scalar.pair.quotient[j].value=model.pair.quotient[j].value;for(int d=0;d<2;++d)for(int j=0;j<3;++j)scalar.poles[d][j].value=model.poles[d][j].value;
  if(!prepare_series(scalar,gate,series))return;
  std::array<std::array<double,5>,4> p{};for(int j=0;j<5;++j)p[0][j]=model.pair.quotient[j].value;p[1]={1,0,1,0,0};for(int d=0;d<2;++d)for(int j=0;j<3;++j)p[2+d][j]=model.poles[d][j].value;
  for(int d=0;d<4;++d){shifted_[d]=translate(p[d],model.pair.mean.value);std::array<double,5> scaled{};double power=1;for(int j=0;j<5;++j){scaled[j]=shifted_[d][j]*power;power*=series.radius;}if(!(scaled[0]>0))return;
   for(int n=0;n<=series.degree;++n){double sum=series.c[n].value;for(int j=1;j<=std::min(n,4);++j)sum-=scaled[j]*divided_[d][n-j];divided_[d][n]=sum/scaled[0];if(!std::isfinite(divided_[d][n]))return;}
  }valid=true;
 }
 double boundary_density(double t)const{const double delta=t-model_->pair.mean.value,x=delta/series.radius;double k=series.c[series.degree].value;for(int n=series.degree-1;n>=0;--n)k=k*x+series.c[n].value;return k*std::sqrt(delta*delta-model_->pair.square.value);}
 std::optional<Jet<N>> integrate(double ta,double tb)const{
  if(!valid)return std::nullopt;const auto& model=*model_;const double r=series.radius,m=model.pair.mean.value,v=model.pair.square.value,xa=(ta-m)/r,xb=(tb-m)/r,e=v/(r*r),h=model.half_width/r;
  if(!(xb>xa&&std::max(std::abs(xa),std::abs(xb))<=1+64*std::numeric_limits<double>::epsilon()&&e<0))return std::nullopt;
  const double sa=std::sqrt(xa*xa-e),sb=std::sqrt(xb*xb-e);std::array<double,19> moment{};moment[0]=std::asinh(xb/h)-std::asinh(xa/h);moment[1]=sb-sa;
  double pa=xa,pb=xb;for(int n=2;n<=series.degree+2;++n){moment[n]=(pb*sb-pa*sa+(n-1)*e*moment[n-2])/n;pa*=xa;pb*=xb;}
  std::array<double,17> weights{};Jet<N> result;double am=0,av=0;
  for(int n=0;n<=series.degree;++n){weights[n]=r*r*(moment[n+2]-e*moment[n]);const double c=series.c[n].value;result.value+=c*weights[n];av-=.5*c*moment[n];am-=c*(r*moment[n+1]+(n?double(n)*r*(moment[n+1]-e*moment[n-1]):0));}
  std::array<std::array<double,5>,4> adjoint{};constexpr double exponent[]={.5,-1.5,-.5,-.5};
  for(int d=0;d<4;++d){double power=1;const int degree=d?2:4;for(int i=0;i<=degree;++i){double a=0;for(int n=i;n<=series.degree;++n)a+=weights[n]*divided_[d][n-i];a*=exponent[d]*power;if(i<degree)am+=a*(i+1)*shifted_[d][i+1];
    for(int k=i;k<=degree;++k){double choose=1;for(int j=1;j<=i;++j)choose*=double(k-j+1)/j;adjoint[d][k]+=a*choose*std::pow(m,k-i);}power*=r;}}
  for(int j=0;j<N;++j){double g=am*model.pair.mean.dp[j]+av*model.pair.square.dp[j]-result.value*model.rho.dp[j]*model.inverse_rho;for(int k=0;k<5;++k)g+=adjoint[0][k]*model.pair.quotient[k].dp[j];for(int d=0;d<2;++d)for(int k=0;k<3;++k)g+=adjoint[d+2][k]*model.poles[d][k].dp[j];if(!std::isfinite(g))return std::nullopt;result.dp[j]=g;}
  if(!(result.value>0&&std::isfinite(result.value)))return std::nullopt;return result;
 }
};
}
