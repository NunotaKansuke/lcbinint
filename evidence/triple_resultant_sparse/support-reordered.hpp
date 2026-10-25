#pragma once
#include "coefficient_ball.hpp"
#include "event_surface.hpp"
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math","fp-contract=off")
#endif
namespace lcbinint::algebraic_boundary::triple::resultant_detail {
// The final polynomial and all root isolation use binary64 intervals.
// Only coefficient formation uses the hardware 64-bit significand tier:
// binary64 coefficient balls cannot separate the nearby discriminant roots.
using OutInterval=surface_detail::I;
using Ball=coefficient_ball_detail::Ball<long double>;using C=coefficient_detail::C<Ball>;
// Before the real determinant, Q has radial degree <=4 and Bezout
// products have degree <=8. Only the final real polynomial needs degree 28.
struct Poly {std::array<C,9>c{};int degree=0;Poly()=default;Poly(double x){c[0]=C{Ball(x)};}Poly(C x){c[0]=x;}};
inline Poly operator-(Poly a){for(int i=0;i<=a.degree;++i)a.c[i]={-a.c[i].re,-a.c[i].im};return a;}
inline Poly operator+(Poly a,const Poly&b){a.degree=std::max(a.degree,b.degree);for(int i=0;i<=b.degree;++i)a.c[i]=a.c[i]+b.c[i];return a;}
inline Poly operator-(Poly a,const Poly&b){return a+(-b);}
inline Poly operator*(const Poly&a,const Poly&b){Poly r;r.degree=a.degree+b.degree;if(r.degree>8)throw std::logic_error("resultant intermediate degree");for(int i=0;i<=a.degree;++i)for(int j=0;j<=b.degree;++j)r.c[i+j]=r.c[i+j]+a.c[i]*b.c[j];return r;}
struct RealPoly {std::array<Ball,29> c{};int degree=0;RealPoly()=default;RealPoly(Ball x){c[0]=x;}};
inline RealPoly operator-(RealPoly a){for(int i=0;i<=a.degree;++i)a.c[i]=-a.c[i];return a;}
inline RealPoly operator+(RealPoly a,const RealPoly&b){a.degree=std::max(a.degree,b.degree);for(int i=0;i<=b.degree;++i)a.c[i]=a.c[i]+b.c[i];return a;}
inline RealPoly operator*(const RealPoly&a,const RealPoly&b){RealPoly r;r.degree=a.degree+b.degree;if(r.degree>28)throw std::logic_error("real determinant degree");for(int i=0;i<=a.degree;++i)if(!a.c[i].zero())for(int j=0;j<=b.degree;++j)if(!b.c[j].zero())r.c[i+j]=r.c[i+j]+a.c[i]*b.c[j];return r;}
using Coefficients=std::array<OutInterval,29>;
inline std::array<Ball,29> support_raw(const Geometry&p,double lo,double hi){
 Poly s;s.degree=1;s.c[1]=C{Ball(1)};
 const Poly a(C{Ball(p.a)}),b(C{Ball(p.bx),Ball(p.by)}),bc(C{Ball(p.bx),-Ball(p.by)}),w(C{Ball(p.X),Ball(p.Y)}),wc(C{Ball(p.X),-Ball(p.Y)});
 const Poly m0(C{Ball(1)-Ball(p.m1)-Ball(p.m2)}),m1(p.m1),m2(p.m2),rho2(C{Ball(p.rho)*Ball(p.rho)});
 // With s=R^2 and z the complex image coordinate:
 // Q = rho^2*s*z*A*B - N*M, where g=N/(s*A), conjugate(g)=M/(z*B).
 // This sextic is R^3 times the boundary Laurent polynomial at z/R.
 std::array<Poly,3>A{s*s,-s*(a+bc),a*bc},B{a*b,-(a+b),Poly(1)};
 std::array<Poly,4>N{},M{};
 for(int i=0;i<3;++i){N[i]=N[i]-s*w*A[i];N[i+1]=N[i+1]+(s-m0)*A[i];M[i]=M[i]+(s-m0)*B[i];M[i+1]=M[i+1]-wc*B[i];}
 N[1]=N[1]-s*s*(m1+m2);N[2]=N[2]+s*(m1*bc+m2*a);M[1]=M[1]+m1*b+m2*a;M[2]=M[2]-m1-m2;
 std::array<Poly,7>q{};for(int i=0;i<4;++i)for(int j=0;j<4;++j)q[i+j]=q[i+j]-N[i]*M[j];
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)q[i+j+1]=q[i+j+1]+rho2*s*A[i]*B[j];
 std::array<Poly,6>u{},v{};for(int i=0;i<6;++i){u[i]=Poly(i+1)*q[i+1];v[i]=Poly(6-i)*q[i];}
 // Bezout(Q_z, 6Q-zQ_z) includes projective infinity correctly.
 // Its determinant is -6^4 times the homogeneous sextic discriminant.
 std::array<std::array<Poly,5>,5>bezout{};
 for(int i=1;i<=5;++i)for(int j=0;j<i;++j){const Poly d=u[i]*v[j]-u[j]*v[i];for(int k=0;k<i-j;++k)bezout[i-1-k][j+k]=bezout[i-1-k][j+k]+d;}
 // B/s^(r_i+r_j) has conjugate reversal symmetry. The constant
 // congruence with paired real/imaginary basis makes it real symmetric;
 // det(U)^2=16. Paired coordinates have identical degree weights.
 const int valuation[5]={2,1,0,0,0},weight[5]={2,3,4,3,2};
 int phase[5][5];for(auto&row:phase)for(auto&x:row)x=-1;
 phase[0][0]=phase[4][0]=phase[1][1]=phase[3][1]=phase[2][2]=0;
 phase[0][4]=phase[1][3]=1;phase[4][4]=phase[3][3]=3;
 std::array<std::array<RealPoly,5>,5>matrix{};
 for(int i=0;i<5;++i)for(int j=i;j<5;++j){
  const int degree=weight[i]+weight[j];RealPoly raw;raw.degree=degree;
  for(int k=0;k<5;++k)if(phase[k][i]>=0)for(int l=0;l<5;++l)if(phase[l][j]>=0){
   const int shift=valuation[k]+valuation[l],ph=(phase[k][i]+phase[l][j])%4;
   for(int n=0;n<=degree;++n){const C z=bezout[k][l].c[n+shift];Ball v=ph==0?z.re:ph==1?-z.im:ph==2?-z.re:z.im;raw.c[n]=raw.c[n]+v;}
  }
  RealPoly current(raw.c[degree]);
  for(int step=1;step<=degree;++step){RealPoly next;next.degree=step;
   for(int k=0;k<=step;++k){if(k<step)next.c[k]=next.c[k]+current.c[k]*Ball(lo);if(k)next.c[k]=next.c[k]+current.c[k-1]*Ball(hi);next.c[k]=next.c[k]+raw.c[degree-step]*Ball(angular_detail::choose(step,k));}current=next;
  }matrix[i][j]=matrix[j][i]=current;
 }
 // Subset determinant: all products contributing to a state have the
 // same weighted degree, so no coefficient-basis conversion is needed.
 std::array<RealPoly,32>det;det[0]=RealPoly(Ball(1));
 // Apply the same row/column permutation: determinant sign is unchanged.
 // Low-degree rows first minimize weighted convolution work (5910 versus
 // 6714 coefficient products for the original order).
 constexpr int order[5]={0,1,4,3,2};
 for(unsigned mask=0;mask<31;++mask){const int row=order[__builtin_popcount(mask)];for(int j=0;j<5;++j)if(!(mask&(1u<<j))){auto term=det[mask]*matrix[row][order[j]];if(__builtin_popcount(mask>>(j+1))%2)term=-term;det[mask|(1u<<j)]=det[mask|(1u<<j)]+term;}}
 std::array<Ball,29> out{};for(int i=0;i<29;++i)out[i]=det[31].c[i];return out;
}
// Direct homogeneous construction avoids forming badly cancelling power
// coefficients of the degree-28 resultant. Divide the binomial scaling only
// after the determinant; subsequent subdivision is ordinary binary64.
inline Coefficients bernstein(const Geometry&p,double lo,double hi){
 const auto raw=support_raw(p,lo,hi);Coefficients out;
 for(int i=0;i<29;++i){using T=typename Ball::Scalar;const T v=T(1)/T(angular_detail::choose(28,i));
  out[i]=(raw[i]*Ball(v,4*std::numeric_limits<T>::epsilon()*std::abs(v))).interval();}
 return out;
}
inline bool excludes(const Coefficients&b){bool pos=true,neg=true;for(auto x:b){if(!x.valid())return false;pos=pos&&x.lo>0;neg=neg&&x.hi<0;if(!pos&&!neg)return false;}return pos||neg;}
inline std::pair<Coefficients,Coefficients> split(Coefficients work){Coefficients left{},right{};left[0]=work[0];right[28]=work[28];for(int k=1;k<=28;++k){for(int j=0;j<=28-k;++j)work[j]=surface_detail::bounded_midpoint(work[j],work[j+1]);left[k]=work[0];right[28-k]=work[28-k];}return {left,right};}
}
#if defined(__GNUC__)
#pragma GCC pop_options
#endif
