#pragma once
#include "root_pair_angular.hpp"
namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail {
// The scalar E/O factor supplies an analytic disk for the same physical
// half-angle map used by the tangent. Gauss-Chebyshev-II 3/5/7-node rules
// integrate degrees 5/9/13; its Cauchy tail controls VALUE error. Parameters
// are evaluated physically at those same nodes, without an eight-lane
// polynomial packet or a completed-primal replay.
template<class Ray,class Endpoint,class Work> std::optional<PairIntegral<8>> cauchy_transport(
 const Ray& ray,double R,const Geometry& p,const Endpoint& lo,const Endpoint& hi,
 const std::array<double,8>& motion,const typename Ray::Core& midpoint,double gate,Work& work,double& error,std::optional<root_pair_detail::Coefficients<0>>* scalar_coefficients=nullptr,std::unique_ptr<root_pair_detail::RadialCache<8>>* canonical_radial=nullptr){
 LCBININT_COUPLED_SCOPE(pair);
 const double width=hi.theta-lo.theta;if(!(gate>=4e-7&&width>0&&lo.error+hi.error<1e-10*width))return std::nullopt;
 const auto required_radius=[&](int n){for(int multiple=2;multiple<=64;++multiple){const double q=1./multiple;double power=1;for(int j=0;j<2*n;++j)power*=q;if(20*power/(1-q)<=gate*.5)return double(multiple);}return 65.;};

#ifdef LCBININT_TRIPLE_CAUCHY_ONLY_THREE
 const double necessary_radius=required_radius(3)*std::tan(width*.25);if(!(necessary_radius<.63))return std::nullopt;
 const auto conjugate=std::conj(midpoint.w);for(auto lens:std::array<Complex,2>{Complex(p.a),Complex(p.bx,p.by)}){const Complex local=lens*conjugate;const double q0=(R-local.real())*(R-local.real())+local.imag()*local.imag(),q2=(R+local.real())*(R+local.real())+local.imag()*local.imag();if(4*R*std::abs(local.imag())*necessary_radius+q2*necessary_radius*necessary_radius>=.39*q0)return std::nullopt;}
 if(scalar_coefficients&&!*scalar_coefficients&&canonical_radial){if(!*canonical_radial)*canonical_radial=std::make_unique<root_pair_detail::RadialCache<8>>(p);root_pair_detail::Coefficients<0> data;data.R=R;data.rho=p.rho;data.lx1=p.a;data.lx2=p.bx;data.ly2=p.by;data.harmonics=(*canonical_radial)->scalar_harmonics(R);scalar_coefficients->emplace(std::move(data));}
#endif
 const double smallest=required_radius(7);if(smallest*std::tan(width*.25)>.63)return std::nullopt;
 root_pair_detail::Arc<0> model;if(!model.initialize(R,p,lo.theta,hi.theta,{},scalar_coefficients,true,nullptr,false))return std::nullopt;
 const double tl=std::tan((lo.theta-model.center)*.5),th=std::tan((hi.theta-model.center)*.5),m=(tl+th)*.5,h=(th-tl)*.5,v=h*h;
 if(!(h>0)||!model.pair.finish(model.polynomial,m,v))return std::nullopt;model.half_width=h;
 int order=0;double multiple=0;
#ifdef LCBININT_TRIPLE_CAUCHY_ONLY_THREE
 const std::array<int,1> orders{3};
#else
 const std::array<int,3> orders{3,5,7};
#endif
 for(int n:orders){const double trial=n==7?smallest:required_radius(n);if(trial*h>.63)continue;if(model.analytic_disk(m,trial*h)){order=n;multiple=trial;break;}}if(!order)return std::nullopt;
 const double predicted=model.phi_from_factor(m,-v),angle=model.center+2*std::atan(m),roundoff=midpoint.roundoff+std::abs(midpoint.angular)*std::abs(angle-midpoint.theta);
 if(!(midpoint.phi>0&&std::abs(predicted-midpoint.phi)+roundoff<1e-9*midpoint.phi))return std::nullopt;
 std::array<double,7> pc{};for(int j=0;j<7;++j)pc[j]=model.polynomial[j].value;const auto residual=root_pair_detail::equations(pc,m,v);
 const double q0=model.pair.quotient[0].value+m*(model.pair.quotient[1].value+m*(model.pair.quotient[2].value+m*(model.pair.quotient[3].value+m*model.pair.quotient[4].value)));
 if(std::abs(residual.e)+h*std::abs(residual.o)>1e-9*v*std::abs(q0))return std::nullopt;
 root_pair_detail::Jet<0> center;if(!model.density(0,center))return std::nullopt;
 const double q=1/multiple,cauchy=std::acos(-1.)*5*std::abs(center.value)*std::pow(q,2*order)/(1-q);
 std::array<double,8> mp{},hp{};for(int j=0;j<8;++j){const double l=.5*(1+tl*tl)*(lo.motion[j]+lo.radial_motion*motion[j]),r=.5*(1+th*th)*(hi.motion[j]+hi.radial_motion*motion[j]);mp[j]=.5*(l+r);hp[j]=.5*(r-l);}
 struct Rule {std::vector<double> x,norm_w;};
 static const auto five=[](){const double pi=std::acos(-1.);return Rule{{std::sqrt(.75),.5,0,-.5,-std::sqrt(.75)},{pi/24,pi/8,pi/6,pi/8,pi/24}};}();
 const auto& base=angular_rule_extended(order==3?2:3);const auto& nodes=order==5?five.x:base.x;const auto& weights=order==5?five.norm_w:base.norm_w;
 const Complex basis(std::cos(model.center),std::sin(model.center));PairIntegral<8> result;
 for(std::size_t k=0;k<nodes.size();++k){const double x=nodes[k],shape=(1-x)*(1+x),t=m+h*x,one=1+t*t,theta=model.center+2*std::atan(t),jac=2*h/one;double phi_error=0;Evaluation<true> jet{};
  if(theta==midpoint.theta){ray.add_parameters(jet,midpoint);phi_error=midpoint.roundoff;}else jet=ray.sample_unit(basis*Complex((1-t*t)/one,2*t/one),theta,phi_error);++work.evaluations;
  std::array<double,8> theta_p{},jac_p{};for(int j=0;j<8;++j){const double tp=mp[j]+hp[j]*x;theta_p[j]=2*tp/one;jac_p[j]=2*hp[j]/one-jac*2*t*tp/one;}
  if(!(jet.phi>0)||phi_error>tangent_phi_roundoff*jet.phi){using Q=local_precision::Q;const auto precise=local_precision::evaluate<true>(Q(R),Q(theta),p);++work.quad_samples;if(!(precise.phi>0))return std::nullopt;const Q root=sqrtq(precise.phi/Q(shape));result.flux+=weights[k]*double(Q(R)*Q(jac)*root);
   for(int j=0;j<8;++j)result.gradient[j]+=weights[k]*double((Q(motion[j])*Q(jac)+Q(R)*Q(jac_p[j]))*root+Q(R)*Q(jac)*(precise.dp[j]+precise.dR*Q(motion[j])+precise.dtheta*Q(theta_p[j]))/(2*root*Q(shape)));
  }else{const double root=std::sqrt(jet.phi/shape);result.flux+=weights[k]*R*jac*root;for(int j=0;j<8;++j)result.gradient[j]+=weights[k]*((motion[j]*jac+R*jac_p[j])*root+R*jac*(jet.dp[j]+jet.dR*motion[j]+jet.dtheta*theta_p[j])/(2*root*shape));}
 }
 if(!std::isfinite(result.flux))return std::nullopt;for(double d:result.gradient)if(!std::isfinite(d))return std::nullopt;
 error=cauchy+1e-7*std::abs(result.flux);return result;
}
}
