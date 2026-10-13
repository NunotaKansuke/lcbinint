#pragma once
#include "lcbinint/magnification/binary_boundary_enclosure.hpp"
namespace lcbinint::magnification::detail {
using VI=math::OutwardInterval;
// Taylor coefficients (derivative/n!), evaluated over a proven graph tube.
struct BoundaryJet {std::array<VI,5> c{};BoundaryJet()=default;BoundaryJet(VI a){c[0]=a;}};
inline BoundaryJet operator+(BoundaryJet a,BoundaryJet b){for(int i=0;i<5;++i)a.c[i]=a.c[i]+b.c[i];return a;}
inline BoundaryJet operator-(BoundaryJet a,BoundaryJet b){for(int i=0;i<5;++i)a.c[i]=a.c[i]-b.c[i];return a;}
inline BoundaryJet operator*(BoundaryJet a,BoundaryJet b){BoundaryJet r;for(int n=0;n<5;++n)for(int i=0;i<=n;++i)r.c[n]=r.c[n]+a.c[i]*b.c[n-i];return r;}
inline BoundaryJet inverse(BoundaryJet a){BoundaryJet r;r.c[0]=VI(1)/a.c[0];for(int n=1;n<5;++n){VI t;for(int i=1;i<=n;++i)t=t+a.c[i]*r.c[n-i];r.c[n]=-t/a.c[0];}return r;}
inline BoundaryJet boundary_g_jet(BoundaryJet x,BoundaryJet y,double s,VI m1,VI m2,double sx,double sy,VI rho2){
 const auto u=x-BoundaryJet(VI(s));const auto d1=inverse(u*u+y*y),d2=inverse(x*x+y*y);
 const auto fx=x-BoundaryJet(m1)*u*d1-BoundaryJet(m2)*x*d2-BoundaryJet(VI(s)*m1+VI(sx));
 const auto fy=y-BoundaryJet(m1)*y*d1-BoundaryJet(m2)*y*d2-BoundaryJet(VI(sy));
 return fx*fx+fy*fy-BoundaryJet(rho2);
}
struct ValidatedHermiteBoundary {bool resolved=false;double fourth_derivative_upper=INFINITY,remainder_upper=INFINITY,area_lower=-INFINITY,area_upper=INFINITY;};
// Conditional local graph certificate; does not establish global image support.
inline ValidatedHermiteBoundary validated_hermite_boundary(double s,double qi,SourcePosition source,double rho,double xl,double xr,double ya,double yb){
 ValidatedHermiteBoundary out;if(!(qi>0))return out;
 for(double v:{s,qi,source.x,source.y,rho,xl,xr,ya,yb})if(!std::isfinite(v))return out;
 if(!critical_enclosure_detail::supported_arithmetic()||rho<=0||xl>=xr||ya>=yb)return out;
 const VI q=std::abs(qi)<1?VI(std::abs(qi)):VI(1)/VI(std::abs(qi));const VI m1=VI(1)/(VI(1)+q),m2=q*m1;
 const double a=qi<1?-std::abs(s):std::abs(s);const VI r2=math::square(VI(rho));
 auto eval=[&](BoundaryJet x,BoundaryJet y){return boundary_g_jet(x,y,a,m1,m2,source.x,source.y,r2);};
 // Mean-value evaluation preserves the narrow variation along each face;
 // raw interval substitution loses cancellations in the lens map.
 const double ym=ya+(yb-ya)*.5;const VI ys(ya,yb),dy=ys-VI(ym);
 auto face=[&](double xx){BoundaryJet yy(ys);yy.c[1]=VI(1);const VI gy=eval(BoundaryJet(VI(xx)),yy).c[1];return eval(BoundaryJet(VI(xx)),BoundaryJet(VI(ym))).c[0]+gy*dy;};
 BoundaryJet probe(VI(xl,xr));probe.c[1]=VI(1);const VI gx_all=eval(probe,BoundaryJet(ys)).c[1];const VI gl=face(xl),gr=face(xr);
 if(!gx_all.finite()||!gl.finite()||!gr.finite()||!((gx_all.lo>0&&gl.hi<0&&gr.lo>0)||(gx_all.hi<0&&gl.lo>0&&gr.hi<0)))return out;
 auto derivatives=[&](VI xs,VI ys){BoundaryJet x(xs),y(ys),probe(xs);probe.c[1]=VI(1);const VI gx=eval(probe,y).c[1];y.c[1]=VI(1);
  if(gx.contains_zero()) {x.c[4]=VI(-INFINITY,INFINITY);return x;}
  for(int n=1;n<5;++n)x.c[n]=-eval(x,y).c[n]/gx;return x;};
 const auto whole=derivatives(VI(xl,xr),VI(ya,yb));if(!whole.c[4].finite())return out;
 auto endpoint=[&](double yy){VI xs(xl,xr);for(int n=0;n<8;++n){const double mid=xs.lo+(xs.hi-xs.lo)*.5;BoundaryJet probe(xs);probe.c[1]=VI(1);const VI gx=eval(probe,BoundaryJet(VI(yy))).c[1];const VI next=VI(mid)-eval(BoundaryJet(VI(mid)),BoundaryJet(VI(yy))).c[0]/gx;xs={std::max(xs.lo,next.lo),std::min(xs.hi,next.hi)};if(!xs.finite()||xs.lo>xs.hi){BoundaryJet invalid;for(auto& c:invalid.c)c=VI(-INFINITY,INFINITY);return invalid;}}return derivatives(xs,VI(yy));};
 const auto left=endpoint(ya),right=endpoint(yb);const VI h=VI(yb)-VI(ya);
 const VI hermite=h*(left.c[0]+right.c[0])/VI(2)+h*h*(left.c[1]-right.c[1])/VI(12);
 // Hermite Peano bound: H^5 max|f''''|/720 = H^5 max|Taylor_4|/30.
 const VI remainder=h*h*h*h*h*VI(whole.c[4].magnitude())/VI(30);
 if(!hermite.finite()||!remainder.finite())return out;
 const VI area=hermite+VI(-remainder.hi,remainder.hi);out.resolved=true;out.fourth_derivative_upper=(VI(24)*VI(whole.c[4].magnitude())).hi;out.remainder_upper=remainder.hi;out.area_lower=area.lo;out.area_upper=area.hi;return out;
}
}
