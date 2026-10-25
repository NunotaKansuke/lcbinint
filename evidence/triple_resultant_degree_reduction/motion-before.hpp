namespace motion_before {
using namespace lcbinint::algebraic_boundary::triple::regular_support_detail;
inline std::optional<std::pair<I,I>> reference(const Disk& disk,double phi,double half,const Geometry&p){
 const auto z=disk.z;const double r=disk.r;auto l=p.lenses();auto m=p.masses();Complex k{},U{},third{};double dk=0,dU=0,dT=0,kc=0,uc=0,tc=0;
 for(int j=0;j<3;++j){double distance=fast_abs(z-l[j]),lower=distance-r;if(!(lower>0))return std::nullopt;auto v=inverse_conjugate(z-l[j]);k+=m[j]*v*v;dk+=2*(m[j]+dm)*r/(lower*lower*lower);kc+=(m[j]+.01)/(lower*lower);if(j){auto weight=m[j]*std::conj(l[j]);U+=weight*v*v;third+=weight*v*v*v;dU+=2*m[j]*fast_abs(l[j])*r/(lower*lower*lower);dT+=3*m[j]*fast_abs(l[j])*r/(lower*lower*lower*lower);uc+=m[j]*fast_abs(l[j])/(lower*lower);tc+=m[j]*fast_abs(l[j])/(lower*lower*lower);}}
 dk+=eps*(1+kc);dU+=eps*(1+uc);dT+=eps*(1+tc);
 const double det=1-std::norm(k),normA=(1+fast_abs(k))/fast_abs(det),q=normA*dk;
 if(!(q<.9))return interval_arc_motion(disk,phi,half,p);
 auto inv=[&](Complex v){return (v-k*std::conj(v))/det;};const Complex phase=std::polar(1.,phi),wp=Complex(0,p.rho)*phase,wpp=-p.rho*phase;
 const double de=2*std::sin(half*.5)+eps;const Complex v=inv(wp),offset=Complex(p.X,p.Y)-U;
 const double ev=normA*(p.rho*de+dk*fast_abs(v))/(1-q)+eps*(1+normA)*(1+fast_abs(v));
 const Complex W=third*std::conj(v);const double eW=dT*(fast_abs(v)+ev)+fast_abs(third)*ev;
 const double g=std::real(std::conj(offset)*wp),gg=std::real(std::conj(offset)*wpp+2.*std::conj(W)*wp);
 const double common=fast_abs(offset)*p.rho*de+dU*p.rho*(1+de);
 const double error=common+eps*(1+fast_abs(g));
 const double error2=common+2*p.rho*((fast_abs(W)+eW)*de+eW)+eps*(1+fast_abs(gg));
 return std::make_pair(I(interval_detail::down(g-error),interval_detail::up(g+error)),I(interval_detail::down(gg-error2),interval_detail::up(gg+error2)));
}
}
