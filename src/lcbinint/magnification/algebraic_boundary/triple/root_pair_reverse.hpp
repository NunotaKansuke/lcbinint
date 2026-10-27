#pragma once
#include "root_pair.hpp"
namespace lcbinint::algebraic_boundary::triple::root_pair_detail {
struct QuotientAdjoint {std::array<double,7> polynomial{};double mean=0,square=0;};
template<int N> QuotientAdjoint quotient_adjoint(const Arc<N>& model,double mean,double square,std::array<double,5> q){
 QuotientAdjoint out;out.mean=mean;out.square=square;const double m=model.pair.mean.value,v=model.pair.square.value,c=m*m-v;
 for(int k=0;k<=2;++k){const double a=q[k];out.polynomial[k+2]+=a;out.mean+=2*a*(model.pair.quotient[k+1].value-m*model.pair.quotient[k+2].value);out.square+=a*model.pair.quotient[k+2].value;q[k+1]+=2*m*a;q[k+2]-=c*a;}
 out.polynomial[5]+=q[3];out.mean+=2*q[3]*model.pair.quotient[4].value;q[4]+=2*m*q[3];out.polynomial[6]+=q[4];return out;
}
// Transpose of the implicit E/O factorization. A scalar integrated local
// sensitivity packet is pulled through one two-by-two solve, rather than
// transporting the same factorization separately for eight parameter RHS.
template<int N> std::optional<std::array<double,7>> factorization_adjoint(
 const Arc<N>& model,double mean_adjoint,double square_adjoint,std::array<double,5> quotient_adjoint){
 const double m=model.pair.mean.value,v=model.pair.square.value,c=m*m-v;
 std::array<double,7> polynomial{},adjoint{};for(int k=0;k<7;++k)polynomial[k]=model.polynomial[k].value;
 for(int k=0;k<=2;++k){const double a=quotient_adjoint[k];adjoint[k+2]+=a;
  mean_adjoint+=2*a*(model.pair.quotient[k+1].value-m*model.pair.quotient[k+2].value);square_adjoint+=a*model.pair.quotient[k+2].value;
  quotient_adjoint[k+1]+=2*m*a;quotient_adjoint[k+2]-=c*a;
 }
 adjoint[5]+=quotient_adjoint[3];mean_adjoint+=2*quotient_adjoint[3]*model.pair.quotient[4].value;
 quotient_adjoint[4]+=2*m*quotient_adjoint[3];adjoint[6]+=quotient_adjoint[4];
 const auto f=equations(polynomial,m,v);const double determinant=f.em*f.ov-f.ev*f.om;
 if(!(std::abs(determinant)>1e-300)||!std::isfinite(determinant))return std::nullopt;
 const double e=(f.ov*mean_adjoint-f.om*square_adjoint)/determinant,o=(f.em*square_adjoint-f.ev*mean_adjoint)/determinant;
 double even=1,odd=0;for(int k=0;k<7;++k){adjoint[k]-=e*even+o*odd;const double next=m*even+v*odd;odd=even+m*odd;even=next;if(!std::isfinite(adjoint[k]))return std::nullopt;}
 return adjoint;
}
inline std::array<double,8> coefficient_adjoint(const Coefficients<8>& data,double basis,const std::array<double,7>& a,const std::array<std::array<double,3>,2>& poles,double rho_adjoint){
 std::array<double,7> h{a[0]+3*a[2]+3*a[4]+a[6],2*(a[0]+a[2]-a[4]-a[6]),-4*a[1]-8*a[3]-4*a[5],2*a[0]-10*a[2]-10*a[4]+2*a[6],-8*a[1]+8*a[5],2*a[0]-30*a[2]+30*a[4]-2*a[6],-12*a[1]+40*a[3]-12*a[5]};
 const double co=std::cos(basis),si=std::sin(basis),c2=co*co-si*si,s2=2*co*si,c3=c2*co-s2*si,s3=s2*co+c2*si;const std::array<double,3> c{co,c2,c3},s{si,s2,s3};
 for(int j=0;j<3;++j){const double r=h[2*j+1],i=h[2*j+2];h[2*j+1]=c[j]*r+s[j]*i;h[2*j+2]=-s[j]*r+c[j]*i;}
 const double R=data.R.value;const std::array<double,2> x{co*data.lx1.value,co*data.lx2.value+si*data.ly2.value},y{-si*data.lx1.value,co*data.ly2.value-si*data.lx2.value};
 double radial=0;std::array<double,2> ax{},ay{};for(int d=0;d<2;++d){const auto& b=poles[d];radial+=2*(R-x[d])*b[0]-4*y[d]*b[1]+2*(R+x[d])*b[2];ax[d]=-2*(R-x[d])*b[0]+2*(R+x[d])*b[2];ay[d]=2*y[d]*(b[0]+b[2])-4*R*b[1];}
 const double l1=co*ax[0]-si*ay[0],l2=co*ax[1]-si*ay[1],ly2=si*ax[1]+co*ay[1];std::array<double,8> result{};
 for(int k=0;k<8;++k){result[k]=radial*data.R.dp[k]+l1*data.lx1.dp[k]+l2*data.lx2.dp[k]+ly2*data.ly2.dp[k]+rho_adjoint*data.rho.dp[k];for(int d=0;d<7;++d)result[k]+=h[d]*data.harmonics[d].dp[k];}return result;
}

}
