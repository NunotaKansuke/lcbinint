#pragma once
#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
namespace lcbinint::holonomic {
// Exact degree-six radial polynomials, assembled once in extended precision.
// This is a count-only representation; local fold Newton keeps its original
// expressions. Uncertain count decisions return to those original expressions.
struct RadialCountPolynomial {
 using Poly=std::array<long double,7>;
 std::array<Poly,5> coefficients{};
 explicit RadialCountPolynomial(const PrimaryFrame& pf) {
  const long double a=pf.a,m=pf.m0,X=pf.X,Y=pf.Y,rho=pf.rho;
  Poly n0r{},n0i{},n1r{},n1i{},n2r{};
  n0r[2]=-X;n0i[2]=-Y;n1r[1]=-1+a*X;n1r[3]=1;n1i[1]=a*Y;n2r[0]=a*m;n2r[2]=-a;
  std::array<Poly,3> re{},im{};
  for(int j=0;j<7;++j){
   re[0][j]=n0r[j]+n1r[j]+n2r[j];im[0][j]=n0i[j]+n1i[j];
   re[1][j]=2*n0i[j];im[1][j]=2*(n2r[j]-n0r[j]);
   re[2][j]=n1r[j]-n0r[j]-n2r[j];im[2][j]=n1i[j]-n0i[j];
  }
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)
   for(int k=0;k<=3;++k)for(int l=0;l<=3;++l)
    coefficients[i+j][k+l]-=re[i][k]*re[j][l]+im[i][k]*im[j][l];
  const long double rr=rho*rho;
  coefficients[0][2]+=rr*a*a;coefficients[0][3]-=2*rr*a;coefficients[0][4]+=rr;
  coefficients[2][2]+=2*rr*a*a;coefficients[2][4]+=2*rr;
  coefficients[4][2]+=rr*a*a;coefficients[4][3]+=2*rr*a;coefficients[4][4]+=rr;
 }
 std::array<double,5> evaluate(double R) const {
  std::array<double,5> out{};
  for(int i=0;i<5;++i){long double v=coefficients[i][6];for(int k=5;k>=0;--k)v=v*(long double)R+coefficients[i][k];out[i]=(double)v;}
  return out;
 }
};
}
