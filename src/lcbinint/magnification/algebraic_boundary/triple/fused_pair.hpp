#pragma once
#include "coupled_profile.hpp"
#include "root_pair.hpp"
#include "radial_coefficient_frame.hpp"
namespace lcbinint::algebraic_boundary::triple::fused_detail {
// A radial spatial jet and eight material parameter RHS. Only the gradient
// integrator instantiates this packet; all RHS share the same pair Jacobian.
struct Mixed {
 double value=0,r=0;std::array<double,8> p{},rp{};
 Mixed()=default;Mixed(double v):value(v){}
};
inline Mixed operator+(Mixed a,const Mixed& b){a.value+=b.value;a.r+=b.r;for(int j=0;j<8;++j){a.p[j]+=b.p[j];a.rp[j]+=b.rp[j];}return a;}
inline Mixed operator-(Mixed a){a.value=-a.value;a.r=-a.r;for(int j=0;j<8;++j){a.p[j]=-a.p[j];a.rp[j]=-a.rp[j];}return a;}
inline Mixed operator-(Mixed a,const Mixed& b){return a+(-b);}
inline Mixed operator*(const Mixed& a,const Mixed& b){Mixed c(a.value*b.value);c.r=a.r*b.value+a.value*b.r;for(int j=0;j<8;++j){c.p[j]=a.p[j]*b.value+a.value*b.p[j];c.rp[j]=a.rp[j]*b.value+a.p[j]*b.r+a.r*b.p[j]+a.value*b.rp[j];}return c;}
inline Mixed compose(const Mixed& a,double v,double d,double dd){Mixed b(v);b.r=d*a.r;for(int j=0;j<8;++j){b.p[j]=d*a.p[j];b.rp[j]=d*a.rp[j]+dd*a.r*a.p[j];}return b;}
inline Mixed inverse(const Mixed& a){double v=1/a.value;return compose(a,v,-v*v,2*v*v*v);}
inline Mixed root(const Mixed& a){double v=std::sqrt(a.value);return compose(a,v,.5/v,-.25/(v*v*v));}
inline Mixed arctangent(const Mixed& a){double d=1/(1+a.value*a.value);return compose(a,std::atan(a.value),d,-2*a.value*d*d);}
struct MixedPair {
 using C=coefficient_detail::C<Mixed>;
 Mixed R,rho,mean,square;std::array<Mixed,5> quotient;std::array<std::array<Mixed,3>,2> poles;
 int minimum_level=6;
 bool initialize(double radius,const Geometry& p,double lo,double hi,const std::array<double,8>& velocity,const std::array<double,8>& slope,root_pair_detail::PreparedFrames* frames=nullptr,double complex_scale=0,bool real_gap=false){
  LCBININT_COUPLED_SCOPE(pair);
  root_pair_detail::Arc<0> primal;const double width=hi-lo;if(!(width>0&&width<1.9*std::acos(-1.)))return false;primal.center=(lo+hi)*.5;
  R=Mixed(radius);R.r=1;R.p=velocity;R.rp=slope;rho=Mixed(p.rho);rho.p[2]=1;
  Mixed X(p.X),Y(p.Y),m1(p.m1),m2(p.m2),a(p.a),bx(p.bx),by(p.by);X.p[0]=1;Y.p[1]=1;m1.p[3]=1;m2.p[4]=1;a.p[5]=1;bx.p[6]=1;by.p[7]=1;
  const double co=std::cos(primal.center),si=std::sin(primal.center);
  const Mixed x1=Mixed(co)*a,y1=Mixed(-si)*a,x2=Mixed(co)*bx+Mixed(si)*by,y2=Mixed(co)*by-Mixed(si)*bx;
  // Construct directly in the frozen, arc-centred frame. This is the same
  // Fourier sextic builder as the binary-inspired first-order pair kernel.
  const Mixed sx=Mixed(co)*X+Mixed(si)*Y,sy=Mixed(co)*Y-Mixed(si)*X;
  const auto frame=frames?frames->get(p,primal.center,radius):nullptr;
  if(frame)primal.center=frame->basis;
  auto pc=frame?frame->at<Mixed>(radius,velocity,slope):root_pair_detail::spectrum_coefficients(R,std::array<C,3>{C{},C{x1,y1},C{x2,y2}},std::array<Mixed,3>{Mixed(1)-m1-m2,m1,m2},C{sx,sy},rho);
  std::array<double,7> values{};for(int j=0;j<7;++j)values[j]=pc[j].value;
  // Solve the very coefficient packet whose RHS will be transported.
  // Independently rotated packets can differ by roundoff in a thin pair.
  std::array<root_pair_detail::Jet<0>,7> scalar;for(int j=0;j<7;++j)scalar[j].value=values[j];
  if(complex_scale>0){const double seed=std::tan(complex_scale*.5);if(!primal.pair.initialize_complex(scalar,0,-seed*seed))return false;}
  else if(!primal.pair.initialize(scalar,std::tan((lo-primal.center)*.5),std::tan((hi-primal.center)*.5)))return false;
  const auto J=root_pair_detail::equations(values,primal.pair.mean.value,primal.pair.square.value);
  const double det=J.em*J.ov-J.ev*J.om;if(!(std::abs(det)>1e-300)||!std::isfinite(det))return false;
  mean=Mixed(primal.pair.mean.value);square=Mixed(primal.pair.square.value);
  // The same two-by-two Jacobian solves every RHS. Form its mixed
  // right-hand side analytically; shifting eighteen-component polynomials
  // twice only to extract these two residuals is unnecessary.
  const double inv=1/det;
  const auto solve=[&](double e,double o){return std::array<double,2>{(J.ev*o-J.ov*e)*inv,(J.om*e-J.em*o)*inv};};
  std::array<double,7> radial{};for(int k=0;k<7;++k)radial[k]=pc[k].r;
  const auto fr=root_pair_detail::equations(radial,mean.value,square.value);
  auto d=solve(fr.e,fr.o);mean.r=d[0];square.r=d[1];
  const auto shifted=root_pair_detail::shifted(values,mean.value);const double v=square.value;
  const double emm=2*shifted[2]+12*v*shifted[4]+30*v*v*shifted[6],emv=3*shifted[3]+10*v*shifted[5],evv=2*shifted[4]+6*v*shifted[6];
  const double omm=6*shifted[3]+20*v*shifted[5],omv=4*shifted[4]+12*v*shifted[6],ovv=2*shifted[5];
  for(int j=0;j<8;++j){
   std::array<double,7> parameter{},mixed{};for(int k=0;k<7;++k){parameter[k]=pc[k].p[j];mixed[k]=pc[k].rp[j];}
   const auto fp=root_pair_detail::equations(parameter,mean.value,v),fm=root_pair_detail::equations(mixed,mean.value,v);
   d=solve(fp.e,fp.o);mean.p[j]=d[0];square.p[j]=d[1];
   const double cross=mean.r*square.p[j]+square.r*mean.p[j];
   const double re=fm.e+fr.em*mean.p[j]+fr.ev*square.p[j]+fp.em*mean.r+fp.ev*square.r+emm*mean.r*mean.p[j]+emv*cross+evv*square.r*square.p[j];
   const double ro=fm.o+fr.om*mean.p[j]+fr.ov*square.p[j]+fp.om*mean.r+fp.ov*square.r+omm*mean.r*mean.p[j]+omv*cross+ovv*square.r*square.p[j];
   d=solve(re,ro);mean.rp[j]=d[0];square.rp[j]=d[1];
  }
  const Mixed constant=mean*mean-square,two_m=Mixed(2)*mean;
  quotient[4]=pc[6];quotient[3]=pc[5]+two_m*quotient[4];for(int j=2;j>=0;--j)quotient[j]=pc[j+2]+two_m*quotient[j+1]-constant*quotient[j+2];
  const auto pole=[&](Mixed x,Mixed y){return std::array<Mixed,3>{(R-x)*(R-x)+y*y,Mixed(-4)*R*y,(R+x)*(R+x)+y*y};};
  if(frame){const auto convert=[](const root_pair_detail::Jet<8>& j){Mixed m(j.value);m.p=j.dp;return m;};poles={pole(convert(frame->lx1),convert(frame->ly1)),pole(convert(frame->lx2),convert(frame->ly2))};}
  else poles={pole(x1,y1),pole(x2,y2)};
  for(int i=0;i<2;++i)for(int j=0;j<3;++j)primal.poles[i][j].value=poles[i][j].value;
  minimum_level=(complex_scale>0||real_gap)?3:primal.short_level();
  // Failure leaves the physical endpoint kernel and its local precision
  // fallback available. Do not accept an unverified polynomial branch.
  if(complex_scale<=0&&!real_gap)for(double x:{-.75,0.,.75}){Mixed q;if(!density(x,q))return false;}
  return true;
 }
 Mixed uniform()const{const auto h=root(square),den=Mixed(1)+mean*mean-square;Mixed width;if(std::abs(den.value)>.25){width=Mixed(2)*arctangent(Mixed(2)*h*inverse(den));if(width.value<0)width.value+=2*std::acos(-1.);}else width=Mixed(2)*(arctangent(mean+h)-arctangent(mean-h));return R*width;}
 // The critical pair is transported together with its angular map.
 // t=m+h*sinh(u), h=sqrt(-v), cancels the root's sqrt((t-m)^2-v)
 // against dt/du before taking any material or mixed derivatives.
 bool critical_density(const Mixed& u,Mixed& out,int side=0)const{
  if(!(square.value<0||square.value>0&&side!=0))return false;
  const auto sh=compose(u,std::sinh(u.value),std::cosh(u.value),std::sinh(u.value));
  const auto ch=compose(u,std::cosh(u.value),std::sinh(u.value),std::cosh(u.value));
  const auto h=root(side?square:-square),t=side?mean+Mixed(side)*h*ch:mean+h*sh,one=Mixed(1)+t*t;
  auto q=quotient[4];for(int j=3;j>=0;--j)q=q*t+quotient[j];
  auto den=one*one*one;for(const auto& pole:poles)den=den*((pole[2]*t+pole[1])*t+pole[0]);
  const auto ratio=q*inverse(den);if(!(ratio.value>0)||!std::isfinite(ratio.value))return false;
  out=Mixed(2)*inverse(rho)*(side?square*sh*sh:(-square)*ch*ch)*root(ratio);
  if(!std::isfinite(out.value)||!std::isfinite(out.r))return false;
  for(int j=0;j<8;++j)if(!std::isfinite(out.p[j])||!std::isfinite(out.rp[j]))return false;return true;
 }
 Mixed critical_coordinate(const Mixed& theta,double basis,int side=0)const{
  const auto angle=(theta-Mixed(basis))*Mixed(.5);const double tangent=std::tan(angle.value),sec=1+tangent*tangent;
  const auto t=compose(angle,tangent,sec,2*tangent*sec),z=side?Mixed(side)*(t-mean)*inverse(root(square)):(t-mean)*inverse(root(-square));
  if(side){const double d=1/std::sqrt(z.value*z.value-1);return compose(z,std::acosh(z.value),d,-z.value*d*d*d);}
  const double d=1/std::sqrt(1+z.value*z.value);return compose(z,std::asinh(z.value),d,-z.value*d*d*d);
 }
 bool density(double x,Mixed& out)const{
  // Differentiate the log of the factored scalar density. All eight RHS
  // share the scalar polynomial, its angle derivatives, and denominators.
  const double v=square.value,h=std::sqrt(v),t=mean.value+h*x;
  const double tr=mean.r+.5*x*square.r/h;
  const auto polynomial=[&](const Mixed* c,int degree,double& value,double& dt,double& dtt,double& radial,double& radial_t){
   value=c[degree].value;dt=dtt=0;radial=c[degree].r;radial_t=0;
   for(int k=degree-1;k>=0;--k){dtt=dtt*t+2*dt;dt=dt*t+value;value=value*t+c[k].value;radial_t=radial_t*t+radial;radial=radial*t+c[k].r;}
  };
  double q,qt,qtt,qr,qrt;polynomial(quotient.data(),4,q,qt,qtt,qr,qrt);
  std::array<double,2> b{},bt{},btt{},br{},brt{};
  for(int i=0;i<2;++i)polynomial(poles[i].data(),2,b[i],bt[i],btt[i],br[i],brt[i]);
  const double one=1+t*t,ratio=-q/(one*one*one*b[0]*b[1]);
  if(!(ratio>0)||!std::isfinite(ratio))return false;
  const double K=2*std::sqrt(ratio)/rho.value,iq=1/q,io=1/one;
  const double qR=(qr+qt*tr)*iq,aR=2*t*tr*io;
  const std::array<double,2> ib{1/b[0],1/b[1]},bR{(br[0]+bt[0]*tr)/b[0],(br[1]+bt[1]*tr)/b[1]};
  const double lr=-rho.r/rho.value+.5*(qR-3*aR-bR[0]-bR[1]);
  out.value=K*v;out.r=K*(square.r+v*lr);
  for(int j=0;j<8;++j){
   const double tp=mean.p[j]+.5*x*square.p[j]/h;
   const double trp=mean.rp[j]+.5*x*(square.rp[j]/h-.5*square.r*square.p[j]/(h*h*h));
   double qp=quotient[4].p[j],qpt=0,qrp=quotient[4].rp[j];
   for(int k=3;k>=0;--k){qpt=qpt*t+qp;qp=qp*t+quotient[k].p[j];qrp=qrp*t+quotient[k].rp[j];}
   const double qP=(qp+qt*tp)*iq;
   const double qRP=(qrp+qrt*tp+qpt*tr+qtt*tr*tp+qt*trp)*iq-qR*qP;
   const double aP=2*t*tp*io,aRP=2*(tr*tp+t*trp)*io-aR*aP;
   double denominator_p=3*aP,denominator_rp=3*aRP;
   for(int i=0;i<2;++i){const auto& c=poles[i];
    const double bp=(c[2].p[j]*t+c[1].p[j])*t+c[0].p[j],bpt=2*c[2].p[j]*t+c[1].p[j];
    const double brp=(c[2].rp[j]*t+c[1].rp[j])*t+c[0].rp[j];
    const double bP=(bp+bt[i]*tp)*ib[i];
    denominator_p+=bP;denominator_rp+=(brp+brt[i]*tp+bpt*tr+btt[i]*tr*tp+bt[i]*trp)*ib[i]-bR[i]*bP;
   }
   const double lp=-rho.p[j]/rho.value+.5*(qP-denominator_p);
   const double lrp=-rho.rp[j]/rho.value+rho.r*rho.p[j]/(rho.value*rho.value)+.5*(qRP-denominator_rp);
   out.p[j]=K*(square.p[j]+v*lp);
   out.rp[j]=K*(square.rp[j]+square.r*lp+square.p[j]*lr+v*(lrp+lr*lp));
  }
  if(!std::isfinite(out.value)||!std::isfinite(out.r))return false;
  for(int j=0;j<8;++j)if(!std::isfinite(out.p[j])||!std::isfinite(out.rp[j]))return false;
  return true;
 }
};
}
