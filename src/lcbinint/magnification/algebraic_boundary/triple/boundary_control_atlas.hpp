#pragma once
#include "radial_events.hpp"
#include "fused_radius.hpp"
#include "local_precision.hpp"
namespace lcbinint::algebraic_boundary::triple {
// A periodic image-boundary primitive. Its signed sum is continuous at
// real folds and across angular chart cuts; its total radial derivative
// therefore integrates to zero over a complete image boundary.
class BoundaryControlAtlas {
public:
 struct Event {double radius=0;Complex unit{},position{};std::array<double,8> velocity{};double cutoff=0,inverse_cutoff_square=0;};
 struct Sample {const Event* event=nullptr;double primitive=0,radial=0,angular_velocity=0;};
private:
 std::vector<Event> events_;bool valid_=false;
 void prepare(){
  if(events_.size()<2||events_.size()>64)return;
  for(auto& e:events_){double nearest=INFINITY;for(const auto& other:events_)if(&e!=&other)nearest=std::min(nearest,std::norm(e.position-other.position));
   e.cutoff=std::min(.45*std::sqrt(nearest),e.radius*.5);if(!(e.cutoff>1e-12*(1+e.radius)&&std::isfinite(e.cutoff)))return;e.inverse_cutoff_square=1/(e.cutoff*e.cutoff);
   for(double v:e.velocity)if(!std::isfinite(v)||std::abs(v)>1e12*(1+e.radius))return;
  }valid_=true;
 }
public:
 explicit BoundaryControlAtlas(std::vector<Event> events):events_(std::move(events)){prepare();}
 BoundaryControlAtlas(const Geometry& p,const std::vector<RadialEvent>& events){
  const double pi=std::acos(-1.);for(const auto& input:events){Event e;e.radius=input.radius;const double theta=(input.reciprocal?pi:0)+2*std::atan(input.coordinate);e.unit={std::cos(theta),std::sin(theta)};e.position=e.radius*e.unit;
   const auto f=fused_detail::Ray<double>(e.radius,p).sample(theta);const double condition=std::max(1.,std::abs(e.radius*f.r));
   if(std::isfinite(f.r)&&std::abs(f.r)>1e-12*condition&&f.roundoff<1e-7&&std::abs(f.phi)<1e-6&&std::abs(f.t)<1e-6*condition){for(int k=0;k<8;++k)e.velocity[k]=-f.p[k]/f.r;}
   else{using Q=local_precision::Q;const Q angle=(input.reciprocal?acosq(Q(-1)):Q(0))+2*atanq(Q(input.coordinate));const auto q=local_precision::evaluate<true>(Q(e.radius),angle,p);if(q.dR==0)return;for(int k=0;k<8;++k)e.velocity[k]=double(-q.dp[k]/q.dR);}
   events_.push_back(std::move(e));
  }prepare();
 }
 bool valid()const{return valid_;}
 Sample sample(double R,Complex unit)const{
  if(!valid_)return {};const Complex position=R*unit;
  for(const auto& e:events_){if(std::abs(R-e.radius)>=e.cutoff)continue;const double distance=std::norm(position-e.position),s=distance*e.inverse_cutoff_square;if(!(s<1))continue;
   const double cosine=unit.real()*e.unit.real()+unit.imag()*e.unit.imag(),sine=unit.imag()*e.unit.real()-unit.real()*e.unit.imag();
   const double b=1-s,b2=b*b,b4=b2*b2,b8=b4*b4,chi=b8*b4,derivative=-12*b8*b2*b;
   const double dr=2*(R-e.radius*cosine)*e.inverse_cutoff_square,dt=2*R*e.radius*sine*e.inverse_cutoff_square;
   return {&e,R*sine*chi,sine*chi+R*sine*derivative*dr,cosine*chi+sine*derivative*dt};
  }return {};
 }
 // Smooth inverse-distance partition. At a critical image the matching
 // event has unit weight and supplies exactly the radial event velocity.
 // Unlike compact bumps this introduces no narrow cutoff into quadrature.
 std::array<double,8> total_derivative(double R,double theta,double theta_r)const{
  std::array<double,8> result{},mean{};if(!valid_)return result;
  const Complex unit(std::cos(theta),std::sin(theta)),position=R*unit;
  double nearest=INFINITY;for(const auto& e:events_)nearest=std::min(nearest,std::norm(position-e.position));
  if(!(nearest>0))return result;
  double sum=0;for(const auto& e:events_){const double d=std::norm(position-e.position),weight=nearest/d;
   const double sine=unit.imag()*e.unit.real()-unit.real()*e.unit.imag();sum+=weight;
   for(int j=0;j<8;++j)mean[j]+=weight*sine*e.velocity[j];}
  for(double& value:mean)value/=sum;
  result=mean;
  for(const auto& e:events_){const double d=std::norm(position-e.position),weight=nearest/d/sum;
   const double cosine=unit.real()*e.unit.real()+unit.imag()*e.unit.imag(),sine=unit.imag()*e.unit.real()-unit.real()*e.unit.imag();
   const double log_derivative=-2*((R-e.radius*cosine)+theta_r*R*e.radius*sine)/d;
   for(int j=0;j<8;++j)result[j]+=R*weight*(theta_r*cosine*e.velocity[j]+log_derivative*(sine*e.velocity[j]-mean[j]));
  }return result;
 }
 Sample sample(double R,double theta)const{return sample(R,Complex(std::cos(theta),std::sin(theta)));}
};
}
