#pragma once
namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail {
// The real endpoint pair leaves a quartic. Narrow interior complex pairs
// appear at negative maxima of that quartic; its derivative is only cubic.
// These are coordinate hints, never support/root-count authority. Ambiguous
// cubic arithmetic or a nearby denominator pole keeps the physical search.
inline std::optional<std::vector<double>> cubic_candidates(std::array<double,4> a){
 double scale=0;for(double x:a)scale=std::max(scale,std::abs(x));if(!(scale>0&&std::isfinite(scale)))return std::nullopt;for(double& x:a)x/=scale;
 const double eps=std::numeric_limits<double>::epsilon();std::vector<double> roots;
 if(a[3]==0){if(a[2]==0){if(a[1]!=0)roots.push_back(-a[0]/a[1]);return roots;}const double d=a[1]*a[1]-4*a[2]*a[0],e=256*eps*(a[1]*a[1]+4*std::abs(a[2]*a[0]));if(std::abs(d)<=e)return std::nullopt;if(d<0)return roots;const double q=-.5*(a[1]+std::copysign(std::sqrt(d),a[1]));roots={q/a[2],a[0]/q};}
 else{if(std::abs(a[3])<1e-12)return std::nullopt;const double b=a[2]/a[3],c=a[1]/a[3],d=a[0]/a[3],p=c-b*b/3,q=2*b*b*b/27-b*c/3+d;
  const double u=q*q*.25,v=p*p*p/27,disc=u+v,error=512*eps*(std::abs(u)+std::abs(v));if(!std::isfinite(disc)||std::abs(disc)<=error)return std::nullopt;
  if(disc>0){const double term=-q*.5-std::copysign(std::sqrt(disc),q),cube=std::cbrt(term);if(cube==0)return std::nullopt;roots.push_back(cube-p/(3*cube)-b/3);}
  else{if(!(p<0))return std::nullopt;const double radius=2*std::sqrt(-p/3),argument=(3*q/(2*p))*std::sqrt(-3/p);if(!(std::abs(argument)<=1+128*eps))return std::nullopt;const double angle=std::acos(std::clamp(argument,-1.,1.))/3,co=std::cos(angle),si=std::sin(angle),center=-b/3;roots={center+radius*co,center+radius*(-.5*co-std::sqrt(.75)*si),center+radius*(-.5*co+std::sqrt(.75)*si)};}
 }
 for(double& r:roots){for(int k=0;k<4;++k){const double f=((a[3]*r+a[2])*r+a[1])*r+a[0],d=(3*a[3]*r+2*a[2])*r+a[1];if(d==0)break;const double step=f/d;if(!std::isfinite(step))return std::nullopt;r-=step;if(std::abs(step)<16*eps*(1+std::abs(r)))break;}const double x=std::abs(r),bound=((std::abs(a[3])*x+std::abs(a[2]))*x+std::abs(a[1]))*x+std::abs(a[0]);if(!std::isfinite(r)||std::abs(((a[3]*r+a[2])*r+a[1])*r+a[0])>512*eps*bound)return std::nullopt;}
 std::sort(roots.begin(),roots.end());for(std::size_t j=1;j<roots.size();++j)if(roots[j]-roots[j-1]<256*eps*(1+std::abs(roots[j])))return std::nullopt;return roots;
}
template<class Ray,class Work> std::optional<AngularChart> quartic_arc_geometry(const Ray& ray,double R,const Geometry& p,double lo,double hi,Work& work,const typename Ray::Core* midpoint,std::optional<root_pair_detail::Coefficients<0>>* scalar_coefficients=nullptr,const std::array<root_pair_detail::Jet<0>,7>* harmonic_values=nullptr){
 if(!midpoint||!(midpoint->phi>0&&midpoint->roundoff<1e-7*midpoint->phi))return std::nullopt;
 if(scalar_coefficients&&!*scalar_coefficients&&harmonic_values){root_pair_detail::Coefficients<0> data;data.R=R;data.rho=p.rho;data.lx1=p.a;data.lx2=p.bx;data.ly2=p.by;data.harmonics=*harmonic_values;scalar_coefficients->emplace(std::move(data));}
 root_pair_detail::Arc<0> model;if(!model.initialize(R,p,lo,hi,{},scalar_coefficients,true,nullptr,false))return std::nullopt;
 const double tl=std::tan((lo-model.center)*.5),th=std::tan((hi-model.center)*.5),m=.5*(tl+th),h=.5*(th-tl),v=h*h;if(!(h>0)||!model.pair.finish(model.polynomial,m,v))return std::nullopt;model.half_width=h;
 const double predicted=model.phi_from_factor(m,-v);if(std::abs(predicted-midpoint->phi)+midpoint->roundoff>1e-7*midpoint->phi)return std::nullopt;
 const auto pole_close=[&](double center,double imaginary){return center>tl&&center<th&&imaginary<.2*(th-tl);};if(pole_close(0,1))return std::nullopt;
 for(const auto& pole:model.poles){const double c= -.5*pole[1].value/pole[2].value,d=pole[0].value/pole[2].value-c*c;if(!(d>0)||pole_close(c,std::sqrt(d)))return std::nullopt;}
 const auto& q=model.pair.quotient;const auto roots=cubic_candidates({q[1].value,2*q[2].value,3*q[3].value,4*q[4].value});if(!roots)return std::nullopt;
 AngularChart result;result.algebraic_hint=true;for(double t:*roots){if(!(t>tl&&t<th))continue;const double second=(12*q[4].value*t+6*q[3].value)*t+2*q[2].value;if(!(second<0))continue;double value=q[4].value,bound=std::abs(value);for(int j=3;j>=0;--j){value=value*t+q[j].value;bound=bound*std::abs(t)+std::abs(q[j].value);}if(std::abs(value)<2048*std::numeric_limits<double>::epsilon()*bound)return std::nullopt;if(!(value<0))continue;const double spread=std::sqrt(2*value/second),theta=model.center+2*std::atan(t),scale=2*spread/(1+t*t);if(!(scale>0&&scale<(hi-lo)*.15&&std::isfinite(scale)))continue;
  double error=0;const auto field=ray.template sample<false>(theta,error);double phi=field.phi;if(!(phi>0)||error>1e-5*phi){const auto exact=local_precision::evaluate<false>(local_precision::Q(R),local_precision::Q(theta),p);phi=double(exact.phi);error=64*std::numeric_limits<double>::epsilon()*std::abs(phi);++work.quad_samples;}if(phi>0)result.minima.push_back({theta,scale,phi,error});
 }return result;
}
}
