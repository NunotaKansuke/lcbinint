#pragma once
#include "physical_jet.hpp"
namespace lcbinint::algebraic_boundary::triple::root_tube_detail {
// A real implicit-function tube t(R)=t0+s(R-R0)+[-radius,radius].
// Uniform nonzero F_t and an inward displacement bound prove one labelled
// angular root throughout the radial interval. Unrelated roots need not be
// polished again at each quadrature node.
struct Tube {
 double origin=0,coordinate=0,slope=0,radius=0,theta=0;bool reciprocal=false,valid=false;
 Tube()=default;
 Tube(const Geometry& p,double a,double b,double R,double angle,double angular_radial,int orientation,double maxradius=.3):origin(R),theta(angle){
  using I=interval_detail::Interval<double>;const double pi=std::acos(-1.);coordinate=std::tan(angle*.5);reciprocal=std::abs(coordinate)>1;if(reciprocal)coordinate=-1/coordinate;slope=.5*(1+coordinate*coordinate)*angular_radial;
  const auto seed=physical_detail::Ray<I>(I(R),p).value(I(coordinate),reciprocal);if(!seed)return;const double residual=std::max(std::abs(seed->lo),std::abs(seed->hi)),span=std::max(R-a,b-R);if(!(span>0&&std::isfinite(residual+slope)))return;
  const I line=I(coordinate)+I(slope)*(I(a,b)-I(origin));
  for(double fraction:{.05,.1,.25,.5,1.})for(int pieces:{1,4,16}){const double width=maxradius*fraction;if(width<256*std::numeric_limits<double>::epsilon()*(1+std::abs(coordinate)))continue;double minimum=INFINITY,velocity=0;bool proved=true;
   for(int piece=0;piece<pieces;++piece){const double ra=a+(b-a)*piece/pieces,rb=a+(b-a)*(piece+1)/pieces;const I local_line=I(coordinate)+I(slope)*(I(ra,rb)-I(origin));const I coordinate_box(local_line.lo-width,local_line.hi+width);if(std::max(std::abs(coordinate_box.lo),std::abs(coordinate_box.hi))>2){proved=false;break;}
    const auto f=physical_detail::physical(I(ra,rb),coordinate_box,p,reciprocal);if(!f||f->t.sign()!=orientation){proved=false;break;}const I derivative=f->r+I(slope)*f->t;const double d=std::min(std::abs(f->t.lo),std::abs(f->t.hi)),motion=std::max(std::abs(derivative.lo),std::abs(derivative.hi));minimum=std::min(minimum,d);const I speed=I(motion)*physical_detail::reciprocal_value(I(d));velocity=std::max(velocity,speed.hi);
   }if(!proved)continue;const I displacement=I(residual)*physical_detail::reciprocal_value(I(minimum))+I(velocity)*I(span);if(displacement.valid()&&displacement.hi<.8*width){radius=width;valid=true;return;}
  }
 }
 double guess(double R)const{const double t=coordinate+slope*(R-origin),base=reciprocal?std::acos(-1.):0;const double angle=base+2*std::atan(t);return theta+std::remainder(angle-theta,2*std::acos(-1.));}
 bool contains(double R,double angle)const{const double base=reciprocal?std::acos(-1.):0,t=std::tan((angle-base)*.5),center=coordinate+slope*(R-origin);return valid&&std::isfinite(t)&&std::abs(t-center)<radius;}
};
}
