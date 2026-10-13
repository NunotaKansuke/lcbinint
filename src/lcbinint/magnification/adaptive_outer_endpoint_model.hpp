#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
namespace lcbinint::magnification::adaptive_detail {
// Diagnostic model only. No certified remainder or endpoint enclosure.
struct OuterEndpointModel {
 bool resolved=false;
 double eta=0, signed_area_residual=0, fourth_relative_defect=0;
 double leading_coefficient=0,next_coefficient=0,squared_cubic_coefficient=0;
};
inline OuterEndpointModel outer_endpoint_quadratic_model(
 const std::array<double,4>& widths,double h,double first_weight,double second_weight,bool integrate_panel=true,double phase_limit=1.5) {
 for(double v:widths) if(!std::isfinite(v)||v<=0) return {};
 if(!std::isfinite(h)||h<=0||!std::isfinite(first_weight)||!std::isfinite(second_weight)) return {};
 const double scale=*std::max_element(widths.begin(),widths.end());
 std::array<double,4> v;
 for(int i=0;i<4;++i) v[i]=(widths[i]/scale)*(widths[i]/scale);
 const double c0=v[0],c2=.5*(v[2]-2*v[1]+v[0]),c1=v[1]-v[0]-c2;
 double eta=std::numeric_limits<double>::infinity();
 const auto offer=[&](double root) {if(std::isfinite(root)&&root<0&&root>-phase_limit) eta=std::min(eta,-root);};
 if(std::abs(c2)<=32*std::numeric_limits<double>::epsilon()*std::max(std::abs(c1),c0)) {
  if(c1!=0) offer(-c0/c1);
 } else {
  const double discriminant=c1*c1-4*c2*c0;
  if(discriminant<0) return {};
  const double q=-.5*(c1+std::copysign(std::sqrt(discriminant),c1));
  if(q==0) return {};
  offer(q/c2);offer(c0/q);
 }
 if(!std::isfinite(eta)) return {};
 const double length=eta+1.5,linear=c1-2*eta*c2;
 if(linear<=0||linear+c2*length<=0) return {};
 static constexpr std::array<double,16> nodes={-0.98940093499164994,-0.9445750230732326,-0.86563120238783176,-0.755404408355003,-0.61787624440264377,-0.45801677765722737,-0.28160355077925892,-0.095012509837637454,0.095012509837637454,0.28160355077925892,0.45801677765722737,0.61787624440264377,0.755404408355003,0.86563120238783176,0.9445750230732326,0.98940093499164994};
 static constexpr std::array<double,16> weights={0.027152459411754037,0.062253523938647706,0.095158511682492591,0.12462897125553403,0.14959598881657676,0.16915651939500262,0.18260341504492361,0.18945061045506859,0.18945061045506859,0.18260341504492361,0.16915651939500262,0.14959598881657676,0.12462897125553403,0.095158511682492591,0.062253523938647706,0.027152459411754037};
 const double end=std::sqrt(length);
 double sum=0;
 if(integrate_panel)for(int i=0;i<16;++i) {
  const double u=.5*end*(nodes[i]+1);
  sum+=weights[i]*2*u*u*std::sqrt(linear+c2*u*u);
 }
 const double integral=h*scale*.5*end*sum;
 const double used=h*(first_weight*widths[0]+second_weight*widths[1]);
 const double predicted=c0+3*c1+9*c2;
 const double defect=std::abs(std::sqrt(std::max(0.,predicted))-widths[3]/scale);
 const double residual=integrate_panel?integral-used:0;
 if(!std::isfinite(residual)||!std::isfinite(defect)) return {};
 return {true,eta,residual,defect,scale*std::sqrt(linear),scale*c2/(2*std::sqrt(linear))};
}
// Cubic interpolation of retained squared widths, with the regular cap root
// selected by continuity from the quadratic model. No lens-map evaluation.
inline OuterEndpointModel outer_endpoint_cubic_model(const std::array<double,4>& widths,double phase_limit=1.5) {
 const auto initial=outer_endpoint_quadratic_model(widths,1,1,1,false);
 if(!initial.resolved)return {};
 const double scale=*std::max_element(widths.begin(),widths.end());
 std::array<double,4> v;for(int k=0;k<4;++k)v[k]=std::pow(widths[k]/scale,2);
 const double d=(v[3]-3*v[2]+3*v[1]-v[0])/6,b=(v[2]-2*v[1]+v[0])/2-3*d,c=v[1]-v[0]-b-d;
 double r=-initial.eta;
 for(int k=0;k<8;++k) {
  const double f=((d*r+b)*r+c)*r+v[0],df=(3*d*r+2*b)*r+c;
  if(!(df>0))return {};
  const double step=f/df;r-=step;
  if(!(r>=-phase_limit&&r<=0))return {};
  if(std::abs(step)<1e-13)break;
 }
 const double a=(3*d*r+2*b)*r+c,curvature=b+3*d*r;
 if(!(a>0)||std::abs(((d*r+b)*r+c)*r+v[0])>1e-10)return {};
 const double eta=-r,L=scale*std::sqrt(a),N=scale*curvature/(2*std::sqrt(a));
 const double expansion=2*std::abs(N)*(eta+3)/L;
 if(!std::isfinite(expansion)||expansion>.5)return {};
 // Retain the quadratic held-row defect as a shape diagnostic. The cubic
 // squared-width coefficient supplies the additional t^(5/2) term.
 return {true,eta,0,initial.fourth_relative_defect,L,N,scale*scale*d};
}

struct ClosedShortRowCap {
 bool resolved=false;
 double integral=0,centre=0,radius=0,amplitude=0;
 double value(double x) const {
  const double u=x-centre;
  return std::abs(u)<radius?amplitude*std::sqrt(std::max(0.,radius*radius-u*u)):0.;
 }
};
// When both endpoints lie within a three-row footprint, the two local
// half-power expansions overlap. Integrate the fitted closed quadratic cap
// itself rather than applying two infinite-lattice endpoint expansions.
inline ClosedShortRowCap closed_short_row_cap(const std::array<double,7>& widths,std::size_t count,int middle=-1) {
 if(count<3||count>7)return {};
 if(middle<0)middle=static_cast<int>(count/2);
 if(middle<=0||static_cast<std::size_t>(middle)>=count-1)return {};
 for(std::size_t k=0;k<count;++k)if(!(widths[k]>0&&std::isfinite(widths[k])))return {};
 const double scale=*std::max_element(widths.begin(),widths.begin()+count);
 const double v0=std::pow(widths[0]/scale,2),v1=std::pow(widths[middle]/scale,2),v2=std::pow(widths[count-1]/scale,2);
 const double last=count-1,m=middle;
 const double b=((v2-v1)/(last-m)-(v1-v0)/m)/last,a=(v1-v0)/m-b*m;
 if(!(b<0))return {};
 const double disc=a*a-4*b*v0;if(!(disc>0))return {};
 const double centre=-a/(2*b),radius=std::sqrt(disc)/(-2*b);
 if(!(centre-radius>=-1.5&&centre-radius<=0&&centre+radius>=last&&centre+radius<=last+1.5))return {};
 const double amplitude=scale*std::sqrt(-b),integral=std::acos(-1.)*amplitude*radius*radius/2;
 if(!std::isfinite(integral))return {};
 return {true,integral,centre,radius,amplitude};
}
inline ClosedShortRowCap closed_three_row_cap(const std::array<double,3>& widths) {
 return closed_short_row_cap({widths[0],widths[1],widths[2],0,0,0,0},3);
}
}
