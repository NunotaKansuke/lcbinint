#pragma once
#include "radial_panel.hpp"
#include "value_trace.hpp"
#include "gradient_angular_chart.hpp"
#include "gradient_radial_chart.hpp"
#include "gradient_hermite.hpp"
#include "root_pair_angular.hpp"
#include "cauchy_transport.hpp"
#include "dyadic_packet_cache.hpp"
#include "ordinary_moment_transport.hpp"

namespace lcbinint::algebraic_boundary::triple {
// First-order physical transport shared by the gradient-owned frontier and
// the retained legacy reference. Value-only calls do not allocate this state.
struct ShapeGradientResult {
    std::array<double,8> uniform{},limb{};
    double uniform_flux=0,limb_flux=0;
    std::size_t evaluations=0,geometry_evaluations=0,quad_samples=0,continued_radii=0,global_radii=0;
    std::size_t reused_gradient_panels=0,reused_gradient_radii=0,geometry_checked_panels=0;
    std::size_t paired_arcs=0,paired_declines=0,paired_uniform_arcs=0,critical_pairs=0,wide_coefficient_packets=0,quad_coefficient_packets=0;
    bool completed=false;
    const char* reason="gradient_replay_incomplete";
};
namespace shape_gradient_detail {
struct Ray {
    struct Core {Complex w{},g{},i1{},i2{},shear{};double norm=0,phi=0,radial=0,angular=0,roundoff=0,theta=0;};
    const Geometry& p;
    double R,inverse_R,inverse_rho2,inverse_rho,parameter_scale,primary,radial,rho2,source_size;
    Ray(const Geometry& geometry,double radius):p(geometry),R(radius),inverse_R(1/R),inverse_rho2(1/(p.rho*p.rho)),inverse_rho(1/p.rho),parameter_scale(-2*inverse_rho2),
        primary((1-p.m1-p.m2)*inverse_R),radial(R-primary),rho2(p.rho*p.rho),source_size(std::hypot(p.X,p.Y)){}
    // A concave phi has no positive interior minimum, hence no complex-pair
    // angular chart is needed. Bound g' and g'' on the whole arc using each
    // pole's minimum distance, rather than inferring this from a few samples.
    bool strictly_concave(double lo,double hi,const Core* retained=nullptr) const {
        const double half=(hi-lo)*.5,mid=lo+half,pad=1e-12;
        if(!(half>0 && half<1))return false;
        const Complex w=retained?retained->w:Complex(std::cos(mid),std::sin(mid)),z=R*w;
        Complex g=retained?retained->g:radial*w-Complex(p.X,p.Y),shear{};
        double first=std::abs(radial),second=first;
        const std::array<Complex,2> lenses{Complex(p.a,0),Complex(p.bx,p.by)};
        const std::array<double,2> masses{p.m1,p.m2};
        for(int k=0;k<2;++k) {
            if(masses[k]==0)continue;
            const Complex delta=z-lenses[k];
            const double distance=std::abs(delta)-R*half-pad*(1+R+std::abs(lenses[k]));
            if(!(distance>0))return false;
            if(!retained){const Complex inverse=inverse_conjugate(delta);
                g-=masses[k]*inverse;shear+=masses[k]*inverse*inverse;}
            const double bound=masses[k]*R/(distance*distance);
            first+=bound;second+=bound*(1+2*R/distance);
        }
        const Complex gt=Complex(0,1)*(retained?z-std::conj(z)*retained->shear:radial*w-std::conj(z)*shear);
        first*=1+pad;second*=1+pad;
        const double lower=(1-pad)*std::abs(gt)-half*second-pad;
        const double upper=(1+pad)*std::abs(g)+half*first+pad;
        return lower>0 && lower*lower>(1+pad)*upper*second;
    }
    // Stationary-point discovery needs only the angular slope. Do not build
    // radial/parameter fields or a phi roundoff certificate for these hints;
    // the final stationary point retains the full physical validation below.
    double angular_slope(double theta)const{
        using C=coefficient_detail::C<double>;using coefficient_detail::conj;
        const auto scale=[](C z,double a){return C(z.re*a,z.im*a);};
        const auto inverse=[&](C z){const double n=z.re*z.re+z.im*z.im;if(n>=32*std::numeric_limits<double>::min()&&std::isfinite(n))return scale(z,1/n);const auto v=inverse_conjugate(Complex(z.re,z.im));return C(v.real(),v.imag());};
        const C w(std::cos(theta),std::sin(theta)),z=scale(w,R),i0=scale(w,inverse_R),i1=inverse(z-C(p.a)),i2=inverse(z-C(p.bx,p.by));
        const C g=scale(w,radial)-C(p.X,p.Y)-scale(i1,p.m1)-scale(i2,p.m2);
        const C first=scale(i1,p.m1)*i1,second=scale(i2,p.m2)*i2,shear=scale(i0,1-p.m1-p.m2)*i0+first+second;
        const C tangent=z-conj(z)*shear;
        return parameter_scale*(-g.re*tangent.im+g.im*tangent.re);
    }
    template<bool Parameters,bool Radial> Evaluation<true> sample_real(Complex unit,double theta,double& phi_error,Core* retained)const{
        using C=coefficient_detail::C<double>;using coefficient_detail::conj;
        const auto scale=[](C a,double b){return C(a.re*b,a.im*b);};
        const auto norm=[](C a){return a.re*a.re+a.im*a.im;};
        const auto inverse=[&](C a){const double n=norm(a);if(n>=32*std::numeric_limits<double>::min()&&std::isfinite(n))return scale(a,1/n);const auto q=inverse_conjugate(Complex(a.re,a.im));return C(q.real(),q.imag());};
        const C w(unit.real(),unit.imag()),z=scale(w,R),i0=scale(w,inverse_R),i1=inverse(z-C(p.a)),i2=inverse(z-C(p.bx,p.by));
        const C g=scale(w,radial)-C(p.X,p.Y)-scale(i1,p.m1)-scale(i2,p.m2);
        const double n=norm(g),eps=std::numeric_limits<double>::epsilon();
        const auto bound=[&](C a){return radius_value_detail::bound_magnitude(Complex(a.re,a.im),norm(a));};
        const double deflection=primary+p.m1*bound(i1)+p.m2*bound(i2);
        const double shear_bound=(1-p.m1-p.m2)/(R*R)+p.m1*norm(i1)+p.m2*norm(i2);
        const double delta=32*eps*(R+source_size+deflection+R*shear_bound);
        phi_error=(2*bound(g)*delta+delta*delta)*inverse_rho2+16*eps*(1+n*inverse_rho2);
        const auto derivative=[&](C dg){return parameter_scale*(g.re*dg.re+g.im*dg.im);};
        const C first=scale(i1,p.m1)*i1,second=scale(i2,p.m2)*i2,shear=scale(i0,1-p.m1-p.m2)*i0+first+second;
        const C tangent=z-conj(z)*shear;
        Evaluation<true> out{};out.phi=1-n*inverse_rho2;out.dtheta=derivative(C(-tangent.im,tangent.re));
        if constexpr(Radial)out.dR=derivative(w+conj(w)*shear);
        if(retained)*retained={unit,Complex(g.re,g.im),Complex(i1.re,i1.im),Complex(i2.re,i2.im),Complex(shear.re,shear.im),n,out.phi,out.dR,out.dtheta,phi_error,theta};
        if constexpr(Parameters)out.dp={derivative(C(-1)),derivative(C(0,-1)),2*n*inverse_rho2*inverse_rho,derivative(i0-i1),derivative(i0-i2),derivative(scale(first,-1)),derivative(scale(second,-1)),derivative(C(-second.im,second.re))};
        return out;
    }
    template<bool Parameters=true,bool Radial=true> Evaluation<true> sample(double theta,double& phi_error,double* curvature=nullptr,Core* retained=nullptr)const {
        return sample_unit<Parameters,Radial>(Complex(std::cos(theta),std::sin(theta)),theta,phi_error,curvature,retained);
    }
    template<bool Parameters=true,bool Radial=true> Evaluation<true> sample_unit(Complex w,double theta,double& phi_error,double* curvature=nullptr,Core* retained=nullptr)const {
        if(!curvature)return sample_real<Parameters,Radial>(w,theta,phi_error,retained);
        const Complex z=R*w;
        const Complex i0=w*inverse_R,i1=inverse_conjugate(z-Complex(p.a,0)),
            i2=inverse_conjugate(z-Complex(p.bx,p.by));
        const Complex g=radial*w-Complex(p.X,p.Y)-p.m1*i1-p.m2*i2;
        const double norm=std::norm(g),eps=std::numeric_limits<double>::epsilon();
        const double deflection=primary+p.m1*radius_value_detail::bound_magnitude(i1,std::norm(i1))+p.m2*radius_value_detail::bound_magnitude(i2,std::norm(i2));
        const double shear_bound=(1-p.m1-p.m2)/(R*R)+p.m1*std::norm(i1)+p.m2*std::norm(i2);
        const double delta=32*eps*(R+source_size+deflection+R*shear_bound);
        phi_error=(2*radius_value_detail::bound_magnitude(g,norm)*delta+delta*delta)*inverse_rho2+16*eps*(1+norm*inverse_rho2);
        const auto derivative=[&](Complex dg){return parameter_scale*(g.real()*dg.real()+g.imag()*dg.imag());};
        Evaluation<true> out{};out.phi=1-norm*inverse_rho2;
        const Complex first_shear=p.m1*i1*i1,second_shear=p.m2*i2*i2;
        const Complex shear=(1-p.m1-p.m2)*i0*i0+first_shear+second_shear;
        if constexpr(Radial)out.dR=derivative(w+std::conj(w)*shear);
        out.dtheta=derivative(Complex(0,1)*(z-std::conj(z)*shear));
        if(curvature) {
            const Complex gt=Complex(0,1)*(z-std::conj(z)*shear);
            const Complex cubic=(1-p.m1-p.m2)*i0*i0*i0+p.m1*i1*i1*i1+p.m2*i2*i2*i2;
            const Complex gtt=-(z+std::conj(z)*shear)+2.*std::conj(z)*std::conj(z)*cubic;
            *curvature=parameter_scale*(std::norm(gt)+std::real(std::conj(g)*gtt));
        }
        if(retained)*retained={w,g,i1,i2,shear,norm,out.phi,out.dR,out.dtheta,phi_error,theta};
        if constexpr(Parameters)out.dp={derivative(-1.),derivative(Complex(0,-1)),2*norm*inverse_rho2*inverse_rho,
            derivative(i0-i1),derivative(i0-i2),derivative(-first_shear),
            derivative(-second_shear),derivative(Complex(-second_shear.imag(),second_shear.real()))};
        return out;
    }
    double short_angular_bound(double lo,double hi,const Core& core,double gate)const{
        const double half=(hi-lo)*.5,D=6*half,eps=std::numeric_limits<double>::epsilon();
        if(!(half>0&&D<1&&core.phi>0&&core.roundoff<gate*.001*core.phi))return INFINITY;
        const double ellipse=6+std::sqrt(35.),bound=4*std::sqrt(5./3)*std::pow(ellipse,-6)/(1-1/ellipse);
        if(bound>gate*.75)return INFINITY;
        const Complex z=R*core.w,zb=std::conj(z),I(0,1),i0=core.w*inverse_R;
        const Complex c0=(1-p.m1-p.m2)*i0*i0*i0,c1=p.m1*core.i1*core.i1*core.i1,c2=p.m2*core.i2*core.i2*core.i2;
        const Complex cubic=c0+c1+c2,quartic=c0*i0+c1*core.i1+c2*core.i2;
        const Complex gt=I*(z-zb*core.shear),gtt=-(z+zb*core.shear)+2.*zb*zb*cubic;
        const Complex gttt=I*(-z+zb*core.shear-6.*zb*zb*cubic+6.*zb*zb*zb*quartic);
        const auto dot=[](Complex a,Complex b){return a.real()*b.real()+a.imag()*b.imag();};
        const double curvature=parameter_scale*(std::norm(gt)+dot(core.g,gtt));if(!(curvature<0))return INFINITY;
        const double third=parameter_scale*(3*dot(gt,gtt)+dot(core.g,gttt));
        const double exponential=std::exp(D),radius_bound=R*exponential,displacement=R*std::expm1(D),base=std::abs(radial)*exponential;
        std::array<double,5> G{base+source_size,base,base,base,base};
        for(auto lens:std::array<std::pair<Complex,double>,2>{{{Complex(p.a),p.m1},{Complex(p.bx,p.by),p.m2}}}){
            const double distance=std::abs(z-lens.first)-displacement;if(!(distance>0))return INFINITY;
            const double inverse=1/distance,a=radius_bound*inverse,a2=a*a,a3=a2*a,a4=a3*a,m=lens.second*inverse;
            G[0]+=m;G[1]+=m*a;G[2]+=m*(a+2*a2);G[3]+=m*(a+6*a2+6*a3);G[4]+=m*(a+14*a2+36*a3+24*a4);
        }
        const auto raw=G;G[0]=std::min(G[0],std::sqrt(core.norm)+D*raw[1]);G[1]=std::min(G[1],std::abs(gt)+D*raw[2]);G[2]=std::min(G[2],std::abs(gtt)+D*raw[3]);G[3]=std::min(G[3],std::abs(gttt)+D*raw[4]);
        const double curvature_error=256*eps*(G[1]*G[1]+G[0]*G[2])*inverse_rho2;
        const double third_error=256*eps*(3*G[1]*G[2]+G[0]*G[3])*inverse_rho2;
        const double fourth_bound=2*(G[0]*G[4]+4*G[1]*G[3]+3*G[2]*G[2])*inverse_rho2;
        const double variation=(std::abs(third)+third_error)*D+.5*fourth_bound*D*D;
        if(!(variation<.25*(-curvature-curvature_error)))return INFINITY;
        return bound;
    }
    void add_parameters(Evaluation<true>& out,const Core& core)const{
        const auto& g=core.g;const auto& i1=core.i1;const auto& i2=core.i2;const Complex i0=core.w*inverse_R;
        const auto derivative=[&](Complex dg){return parameter_scale*(g.real()*dg.real()+g.imag()*dg.imag());};
        out.phi=core.phi;out.dtheta=core.angular;out.dR=derivative(core.w+std::conj(core.w)*core.shear);
        out.dp={derivative(-1.),derivative(Complex(0,-1)),2*core.norm*inverse_rho2*inverse_rho,derivative(i0-i1),derivative(i0-i2),derivative(-p.m1*i1*i1),derivative(-p.m2*i2*i2),derivative(Complex(0,1)*p.m2*i2*i2)};
    }
};
using GeometryCores=std::array<std::array<Ray::Core,3>,3>;
// The value/binary physical continuation already works in a bounded
// half-angle coordinate. Keep all Newton fields rational in that coordinate;
// convert back to an angle once, after the residual has passed the same stop.
template<bool Radial> bool half_angle_polish(const Ray& ray,double& theta,double width,Evaluation<true>& jet,double& error,Ray::Core& core,ShapeGradientResult& work,double endpoint_relative){
 const double initial=theta,pi=std::acos(-1.),eps=std::numeric_limits<double>::epsilon();
 const bool reciprocal=std::abs(std::remainder(initial,2*pi))>pi*.5;const double basis=reciprocal?pi:0,sign=reciprocal?-1:1;
 double coordinate=std::tan(std::remainder(initial-basis,2*pi)*.5);const double origin=coordinate;
 for(int iteration=0;iteration<10;++iteration){
  const double square=coordinate*coordinate,inverse=1/(1+square);const Complex unit(sign*(1-square)*inverse,sign*2*coordinate*inverse);
  auto field=ray.template sample_unit<false,Radial>(unit,0,error,nullptr,&core);++work.geometry_evaluations;
  const double angle_step=field.phi/field.dtheta,step=.5*(1+square)*angle_step;
  if(!std::isfinite(step)||!std::isfinite(angle_step))return false;
  if(std::abs(angle_step)<std::max({32*eps*(1+std::abs(initial)),2*error/std::abs(field.dtheta),endpoint_relative*width})){
   const double candidate=initial+std::remainder(basis+2*std::atan(coordinate)-initial,2*pi);if(std::abs(candidate-initial)>.25*width)return false;
   theta=candidate;core.theta=theta;jet=field;return true;
  }
  const double next=coordinate-step;if(!std::isfinite(next)||std::abs(next)>4)return false;
  if(2*(std::abs(coordinate-origin)+std::abs(step))>.25*width){const double a=initial+std::remainder(basis+2*std::atan(coordinate)-initial,2*pi),b=initial+std::remainder(basis+2*std::atan(next)-initial,2*pi);if(std::abs(a-initial)+std::abs(b-a)>.25*width)return false;}
  coordinate=next;
 }return false;
}
inline double endpoint_relative_budget(double value_gate){
#if defined(LCBININT_TRIPLE_ENDPOINT_ROUNDOFF_BUDGET)
 return std::min(1e-7,value_gate*1e-2);
#elif defined(LCBININT_TRIPLE_VALUE_ENDPOINT_BUDGET)
 return std::min(1e-8,value_gate*1e-3);
#else
 return 0;
#endif
}
struct Endpoint {double theta=0,radial_motion=0,slope=0,error=0;std::array<double,8> motion{};local_precision::Q precise_theta=0;bool wide=false;};
inline void endpoint_derivatives(const Evaluation<true>& jet,Endpoint& out){
 const double inverse=-1/jet.dtheta;out.radial_motion=jet.dR*inverse;for(int j=0;j<8;++j)out.motion[j]=jet.dp[j]*inverse;
 out.slope=jet.dtheta;
}
inline bool endpoint_impl(const Ray& ray,double R,const Geometry& p,double theta,double width,
                     Endpoint& out,ShapeGradientResult& work,PolarMixedJet* mixed=nullptr,bool polished=false,const local_precision::Q* retained_wide=nullptr,const Ray::Core* retained_core=nullptr,double endpoint_relative=0) {
    LCBININT_COUPLED_SCOPE(endpoint);
    const double initial=theta,eps=std::numeric_limits<double>::epsilon();
    bool resolved=false;Evaluation<true> jet{};double roundoff=0;
    if(!mixed&&retained_wide){const auto q=local_precision::evaluate<true>(local_precision::Q(R),*retained_wide,p);++work.quad_samples;++work.evaluations;if(q.dtheta==0)return false;
        out.theta=double(*retained_wide);out.precise_theta=*retained_wide;out.wide=true;out.radial_motion=double(-q.dR/q.dtheta);out.slope=double(q.dtheta);out.error=4*eps*(1+std::abs(out.theta));for(int j=0;j<8;++j)out.motion[j]=double(-q.dp[j]/q.dtheta);return true;}
    if(!mixed&&polished){
        if(retained_core&&retained_core->theta==theta&&retained_core->angular!=0){ray.add_parameters(jet,*retained_core);roundoff=retained_core->roundoff;}
        else jet=ray.sample(theta,roundoff);
        ++work.evaluations;const double step=jet.phi/jet.dtheta;
        if(std::isfinite(step)&&std::abs(step)<=std::max({64*eps*(1+std::abs(theta)),4*roundoff/std::abs(jet.dtheta),endpoint_relative*width})&&roundoff<std::abs(jet.dtheta)*width*1e-7){out.theta=theta;endpoint_derivatives(jet,out);out.error=2*(std::abs(jet.phi)+roundoff)/std::abs(jet.dtheta);return std::all_of(out.motion.begin(),out.motion.end(),[](double x){return std::isfinite(x);});}}
    if(mixed) {
        // Retained primal endpoints usually already satisfy the Newton stop.
        // Evaluate their tangent once instead of repeating a spatial polish.
        *mixed=polar_mixed(R,theta,p);++work.evaluations;
        const double step=mixed->phi/mixed->t;
        if(std::isfinite(step) && std::abs(step)<32*eps*(1+std::abs(theta)) &&
           mixed->roundoff<std::abs(mixed->t)*width*1e-7) {
            out.theta=theta;out.radial_motion=-mixed->r/mixed->t;out.slope=mixed->t;
            for(int j=0;j<8;++j)out.motion[j]=-mixed->p[j]/mixed->t;
            return std::all_of(out.motion.begin(),out.motion.end(),[](double x){return std::isfinite(x);});
        }
    }
    Ray::Core core;
#ifdef LCBININT_TRIPLE_HALF_ANGLE_ENDPOINTS
    if(!mixed)resolved=half_angle_polish<false>(ray,theta,width,jet,roundoff,core,work,endpoint_relative);
#endif
    for(int k=0;!resolved&&k<10;++k) {
        double curvature=0;
#ifdef LCBININT_TRIPLE_HALLEY_ENDPOINTS
        jet=ray.sample<false,false>(theta,roundoff,&curvature,&core);
#else
        jet=ray.sample<false,false>(theta,roundoff,nullptr,&core);
#endif
        ++work.geometry_evaluations;
        const double step=jet.phi/jet.dtheta;
        if(!std::isfinite(step) || std::abs(theta-initial)+std::abs(step)>.25*width)return false;
        if(std::abs(step)<std::max({32*eps*(1+std::abs(theta)),2*roundoff/std::abs(jet.dtheta),endpoint_relative*width})) {resolved=true;break;}
        double update=step;
#ifdef LCBININT_TRIPLE_HALLEY_ENDPOINTS
        const double ratio=.5*step*curvature/jet.dtheta;if(std::abs(ratio)<.25){const double candidate=step/(1-ratio);if(std::abs(theta-initial)+std::abs(candidate)<=.25*width)update=candidate;}
#endif
        theta-=update;
    }
    if(resolved) {
        if(mixed) {
            *mixed=polar_mixed(R,theta,p);roundoff=mixed->roundoff;
            jet.phi=mixed->phi;jet.dR=mixed->r;jet.dtheta=mixed->t;jet.dp=mixed->p;
        } else ray.add_parameters(jet,core);
        ++work.evaluations;
        resolved=std::isfinite(jet.dtheta) && roundoff<std::abs(jet.dtheta)*width*1e-7;
    }
    if(resolved) {
        out.theta=theta;endpoint_derivatives(jet,out);out.error=2*(std::abs(jet.phi)+roundoff)/std::abs(jet.dtheta);
    } else {
        // Hermite replay can retain its primal panel and use the incumbent
        // local repair without computing a promoted mixed jet speculatively.
        if(mixed)return false;
        // Upgrade only the unresolved endpoint; the primal proof and every
        // other radius stay retained. No broad small-arc quad threshold.
        using Q=local_precision::Q;Q refined;
        if(!local_precision::endpoint(R,theta,p,refined))return false;
        const auto q=local_precision::evaluate<true>(Q(R),refined,p);
        ++work.quad_samples;++work.evaluations;
        if(q.dtheta==0 || !finiteq(q.dtheta))return false;
        out.theta=double(refined);out.precise_theta=refined;out.wide=true;out.error=4*eps*(1+std::abs(out.theta));out.radial_motion=double(-q.dR/q.dtheta);out.slope=double(q.dtheta);
        for(int j=0;j<8;++j)out.motion[j]=double(-q.dp[j]/q.dtheta);
    }
    return std::all_of(out.motion.begin(),out.motion.end(),[](double x){return std::isfinite(x);});
}
inline bool endpoint(double R,const Geometry& p,double theta,double width,Endpoint& out,ShapeGradientResult& work,PolarMixedJet* mixed=nullptr){
    const Ray ray(p,R);return endpoint_impl(ray,R,p,theta,width,out,work,mixed);
}

// Geometry-only planning never evaluates parameter RHS or a limb integral.
// Retain a promoted endpoint itself, so adding tangents needs one Q field
// evaluation rather than solving the same promoted root again.
inline bool spatial_endpoint(const Ray& ray,double R,const Geometry& p,double theta,double width,Endpoint& out,ShapeGradientResult& work,Ray::Core* retained=nullptr,double endpoint_relative=0){
    const double initial=theta,eps=std::numeric_limits<double>::epsilon();Evaluation<true> q{};double error=0;bool resolved=false;
#ifdef LCBININT_TRIPLE_HALF_ANGLE_ENDPOINTS
    Ray::Core local;auto* saved=retained?retained:&local;resolved=half_angle_polish<true>(ray,theta,width,q,error,*saved,work,endpoint_relative);
#endif
    for(int k=0;!resolved&&k<10;++k){double curvature=0;
#ifdef LCBININT_TRIPLE_HALLEY_ENDPOINTS
        q=ray.sample<false>(theta,error,&curvature,retained);
#else
        q=ray.sample<false>(theta,error,nullptr,retained);
#endif
        ++work.geometry_evaluations;const double step=q.phi/q.dtheta;if(!std::isfinite(step)||std::abs(theta-initial)+std::abs(step)>width*.25)return false;
        if(std::abs(step)<std::max({32*eps*(1+std::abs(theta)),2*error/std::abs(q.dtheta),endpoint_relative*width})){resolved=true;break;}
        double update=step;
#ifdef LCBININT_TRIPLE_HALLEY_ENDPOINTS
        const double ratio=.5*step*curvature/q.dtheta;if(std::abs(ratio)<.25){const double candidate=step/(1-ratio);if(std::abs(theta-initial)+std::abs(candidate)<=.25*width)update=candidate;}
#endif
        theta-=update;
    }
    if(!resolved)return false;
    if(std::isfinite(q.dtheta)&&error<std::abs(q.dtheta)*width*1e-7){out.theta=theta;out.radial_motion=-q.dR/q.dtheta;out.slope=q.dtheta;out.error=2*(std::abs(q.phi)+error)/std::abs(q.dtheta);return std::isfinite(out.radial_motion);}
    using Q=local_precision::Q;Q thetaq;if(!local_precision::endpoint(R,theta,p,thetaq))return false;const auto precise=fused_detail::Ray<Q>(R,p).sample<false>(thetaq);++work.quad_samples;++work.geometry_evaluations;if(precise.t==0)return false;
    out.theta=double(thetaq);out.precise_theta=thetaq;out.wide=true;out.radial_motion=double(-precise.r/precise.t);out.slope=double(precise.t);out.error=4*eps*(1+std::abs(out.theta));return std::isfinite(out.radial_motion);
}

struct RadiusAnchor {
    double radius=0;
    std::size_t count=0;
    struct Arc {double lo,hi,dlo,dhi;local_precision::Q precise_lo=0,precise_hi=0;bool wide_lo=false,wide_hi=false;};
    std::array<Arc,3> arcs{};bool geometry_verified=false;
};
struct RadiusResult {std::array<double,8> uniform{},limb{};double f0=0,fh=0,endpoint_error=0;};
struct TransportedArc {root_pair_detail::Arc<8> model;double lr=0,hr=0,error=0;bool ready=false;};
inline bool radius(double R,const Geometry& p,const AngularTopology& topology,
                   bool with_limb,int level,const std::array<double,8>& motion,RadiusResult& out,ShapeGradientResult& work,double seed_radius,RadiusAnchor* anchor=nullptr,std::unique_ptr<root_pair_detail::RadialCache<8>>* pair_table=nullptr,double value_gate=1e-6,double* angular_error=nullptr,bool paired_transport=false,const std::vector<TransportedArc>* transported=nullptr,int critical_order=8,const GeometryCores* geometry_cores=nullptr,const point_polynomial_detail::Cache<double>* stationary_cache=nullptr,int selected_arc=-1) {
    LCBININT_COUPLED_SCOPE(angular);
    const bool full=topology.roots.empty();Ray ray(p,R);
    RadiusAnchor next_anchor;next_anchor.radius=R;
    const bool nearby=anchor && anchor->count==topology.arcs.size() &&
        std::abs(R-anchor->radius)<std::abs(R-seed_radius);
    std::array<AngularArc,3> resolved_arcs{};std::size_t resolved_count=0;
    const bool retained_proof=anchor&&anchor->geometry_verified&&anchor->radius==R&&anchor->count==topology.arcs.size();bool all_retained=retained_proof;
    std::optional<root_pair_detail::Coefficients<0>> scalar_coefficients;std::optional<root_pair_detail::Coefficients<8>> coefficients;std::optional<AngularStationarity> stationary;
    for(const auto& arc:topology.arcs) {
        if(selected_arc>=0&&int(next_anchor.count)!=selected_arc){
            if(!retained_proof||!geometry_cores)return false;
            const auto saved=anchor->arcs[next_anchor.count];resolved_arcs[resolved_count++]={saved.lo,saved.hi};next_anchor.arcs[next_anchor.count++]=saved;continue;
        }
        Endpoint lo,hi;lo.theta=arc.lo;hi.theta=arc.hi;Ray::Core midpoint_core;bool have_midpoint=false;
        if(!full) {
            const auto* packet=transported&&next_anchor.count<transported->size()&&(*transported)[next_anchor.count].ready?&(*transported)[next_anchor.count]:nullptr;
            if(packet){const auto& model=packet->model;const auto& pair=model.pair;const double m=pair.mean.value,h=model.half_width,l=m-h,r=m+h;
                lo.theta=model.center+2*std::atan(l);hi.theta=model.center+2*std::atan(r);lo.slope=1;hi.slope=-1;
                for(int j=0;j<8;++j){lo.motion[j]=2*(pair.mean.dp[j]-.5*pair.square.dp[j]/h)/(1+l*l);hi.motion[j]=2*(pair.mean.dp[j]+.5*pair.square.dp[j]/h)/(1+r*r);}
                out.endpoint_error+=R*packet->error;
            }else{
            const double width=arc.hi-arc.lo;double l=arc.lo,h=arc.hi;
            bool predicted=false;
            if(nearby) {
                double best=width*.25;
                for(std::size_t i=0;i<anchor->count;++i) {
                    const auto& a=anchor->arcs[i];
                    const double shift=2*std::acos(-1.)*std::round((arc.lo-a.lo)/(2*std::acos(-1.)));
                    const double dl=a.lo+shift+a.dlo*(R-anchor->radius),dh=a.hi+shift+a.dhi*(R-anchor->radius);
                    const double distance=std::max(std::abs(dl-arc.lo),std::abs(dh-arc.hi));
                    if(distance<best){best=distance;l=dl;h=dh;predicted=true;}
                }
            }
            if(!predicted && seed_radius!=R) {
                Ray seed_ray(p,seed_radius);double unused=0;
                const auto jl=seed_ray.sample<false>(l,unused),jh=seed_ray.sample<false>(h,unused);
                const double dl=-(R-seed_radius)*jl.dR/jl.dtheta,dh=-(R-seed_radius)*jh.dR/jh.dtheta;
                if(std::abs(dl)<width*.25)l+=dl;if(std::abs(dh)<width*.25)h+=dh;
            }
            const bool polished=anchor&&anchor->radius==R&&anchor->count==topology.arcs.size();const auto* frozen=polished?&anchor->arcs[next_anchor.count]:nullptr;
            if(!endpoint_impl(ray,R,p,l,width,lo,work,nullptr,polished,frozen&&frozen->wide_lo?&frozen->precise_lo:nullptr,polished&&geometry_cores?&(*geometry_cores)[next_anchor.count][0]:nullptr,paired_transport?endpoint_relative_budget(value_gate):0)){work.reason="gradient_replay_left_endpoint";return false;}
            if(!endpoint_impl(ray,R,p,h,width,hi,work,nullptr,polished,frozen&&frozen->wide_hi?&frozen->precise_hi:nullptr,polished&&geometry_cores?&(*geometry_cores)[next_anchor.count][1]:nullptr,paired_transport?endpoint_relative_budget(value_gate):0)){work.reason="gradient_replay_right_endpoint";return false;}
            }
            if(!(lo.slope>0 && hi.slope<0)){work.reason="gradient_replay_endpoint_orientation";return false;}
            if(!(hi.theta>lo.theta) || resolved_count==resolved_arcs.size())return false;
            // The retained band supplies the root count. Distinct, disjoint
            // continued arcs must still account for that count: reject a
            // Newton jump onto an already found positive interval.
            const bool retained=retained_proof&&lo.theta==anchor->arcs[resolved_count].lo&&hi.theta==anchor->arcs[resolved_count].hi;all_retained=all_retained&&retained;
            if(!all_retained)for(std::size_t i=0;i<resolved_count;++i)for(int turn:{-1,0,1}) {
                const double shift=turn*2*std::acos(-1.);
                if(std::max(lo.theta,resolved_arcs[i].lo+shift)<std::min(hi.theta,resolved_arcs[i].hi+shift))return false;
            }
            resolved_arcs[resolved_count++]={lo.theta,hi.theta};
            if(retained&&with_limb&&geometry_cores){const auto& core=(*geometry_cores)[resolved_count-1][2];if(core.theta==(lo.theta+hi.theta)*.5&&core.phi>0){midpoint_core=core;have_midpoint=true;}}
            double unused=0;if(!packet&&!retained){const auto middle=ray.sample<false>((lo.theta+hi.theta)*.5,unused,nullptr,with_limb?&midpoint_core:nullptr);have_midpoint=with_limb;if(!(middle.phi>0)){work.reason="gradient_replay_membership";return false;}}
            for(int j=0;j<8;++j)out.uniform[j]+=R*(hi.motion[j]-lo.motion[j]+
                (hi.radial_motion-lo.radial_motion)*motion[j]);
        }
        if(next_anchor.count<next_anchor.arcs.size()){const auto* packet=transported&&next_anchor.count<transported->size()&&(*transported)[next_anchor.count].ready?&(*transported)[next_anchor.count]:nullptr;next_anchor.arcs[next_anchor.count++]={lo.theta,hi.theta,packet?packet->lr:lo.radial_motion,packet?packet->hr:hi.radial_motion,lo.precise_theta,hi.precise_theta,lo.wide,hi.wide};}
        out.endpoint_error+=R*(lo.error+hi.error);
        const double f0=R*(hi.theta-lo.theta);out.f0+=f0;
        for(int j=0;j<8;++j)out.uniform[j]+=motion[j]*f0/R;
        if(!with_limb)continue;
        if(!have_midpoint){double unused=0;ray.sample<false>((lo.theta+hi.theta)*.5,unused,nullptr,&midpoint_core);have_midpoint=true;}
        double short_bound=INFINITY;
#ifdef LCBININT_TRIPLE_PHYSICAL_CAUCHY_SHORT
        if(paired_transport&&!full&&lo.error+hi.error<1e-10*(hi.theta-lo.theta))short_bound=ray.short_angular_bound(lo.theta,hi.theta,midpoint_core,value_gate);
#endif
        const bool short_rule=std::isfinite(short_bound);
        const bool concave=short_rule||(paired_transport&&ray.strictly_concave(lo.theta,hi.theta,&midpoint_core));
        const bool physical_resolved=midpoint_core.phi>0&&midpoint_core.roundoff<1e-7*midpoint_core.phi;

#ifdef LCBININT_TRIPLE_CAUCHY_ORDINARY
        if(paired_transport&&!full&&physical_resolved){double error=0;if(const auto fast=cauchy_transport(ray,R,p,lo,hi,motion,midpoint_core,value_gate,work,error,&scalar_coefficients,pair_table)){out.fh+=fast->flux;for(int j=0;j<8;++j)out.limb[j]+=fast->gradient[j];if(angular_error)*angular_error+=error;continue;}}
#endif
        std::optional<root_pair_detail::Arc<8>> ordinary;
        if(paired_transport&&!full&&!physical_resolved){
            if(pair_table&&!*pair_table)*pair_table=std::make_unique<root_pair_detail::RadialCache<8>>(p);
            root_pair_detail::Arc<8> model;const auto* radial=pair_table?pair_table->get():nullptr;
            const auto* packet=transported&&next_anchor.count>0&&next_anchor.count<=transported->size()&&(*transported)[next_anchor.count-1].ready?&(*transported)[next_anchor.count-1]:nullptr;
            if(packet)ordinary=packet->model;
            else if(model.initialize(R,p,lo.theta,hi.theta,motion,&coefficients,true,radial)){
                const auto& pair=model.pair;const double m=pair.mean.value,v=pair.square.value,h=model.half_width,den=(1+m*m-v)*(1+m*m-v)+4*v;
                bool consistent=true;
                for(int j=0;j<8;++j){const double computed=motion[j]*(hi.theta-lo.theta)+R*(2/h)*((1+m*m+v)*pair.square.dp[j]-4*m*v*pair.mean.dp[j])/den;
                    const double physical=motion[j]*(hi.theta-lo.theta)+R*(hi.motion[j]-lo.motion[j]+(hi.radial_motion-lo.radial_motion)*motion[j]);
                    consistent=consistent&&std::abs(computed-physical)<1e-6*(1+std::abs(physical));}
                double error=0;const double theta=model.center+2*std::atan(m);const auto q=ray.sample<false>(theta,error);const double phi=model.phi_from_factor(m,-v);
                if(consistent&&q.phi>0&&std::abs(phi-q.phi)+error<1e-6*q.phi)ordinary=std::move(model);
            }
            if(ordinary&&ordinary->disk_resolved(1)){double error=0;if(const auto pair=ordinary_arc(*ordinary,value_gate,work,error)){if(angular_error)*angular_error+=error;out.fh+=pair->flux;for(int j=0;j<8;++j)out.limb[j]+=pair->gradient[j];continue;}}
        }
        std::optional<AngularChart> retained;
        std::array<double,8> lower_motion{},upper_motion{};for(int j=0;j<8;++j){lower_motion[j]=lo.motion[j]+lo.radial_motion*motion[j];upper_motion[j]=hi.motion[j]+hi.radial_motion*motion[j];}
        if(!concave&&stationary_cache&&!stationary)stationary.emplace(R,p,*stationary_cache);
        if(!concave)if(const auto pair=critical_arc<8>(ray,R,p,lo.theta,hi.theta,!full,motion,work,coefficients,pair_table,retained,critical_order,paired_transport?&lower_motion:nullptr,paired_transport?&upper_motion:nullptr,stationary?&*stationary:nullptr,&midpoint_core,&scalar_coefficients,value_gate)) {
            ++work.paired_arcs;if(angular_error)*angular_error+=pair->error;out.fh+=pair->flux;for(int j=0;j<8;++j)out.limb[j]+=pair->gradient[j];continue;
        }
        ++work.paired_declines;
        const double half=(hi.theta-lo.theta)*.5,mid=lo.theta+half;
        if(paired_transport&&retained&&retained->nodes.empty())render_angular_chart(*retained,lo.theta,hi.theta,!full,std::max(32,2*critical_order),std::max(16,critical_order));
        if(!concave)if(const auto chart=complex_arc_chart(ray,R,p,lo.theta,hi.theta,!full,motion,work,&retained)) {
            if(!chart->valid){work.reason="gradient_replay_angular_chart";return false;}
            if(angular_error)*angular_error+=1e-6*std::abs(chart->flux);out.fh+=chart->flux;
            for(int j=0;j<8;++j)out.limb[j]+=chart->gradient[j];
            continue;
        }
        if(paired_transport&&!full&&physical_resolved
#ifndef LCBININT_TRIPLE_NONCONCAVE_MOMENTS
            &&concave
#endif
        ){if(const auto packet=ordinary_moment_transport(R,p,lo,hi,motion,midpoint_core,value_gate,work,coefficients,scalar_coefficients,pair_table,concave)){out.fh+=packet->flux;for(int j=0;j<8;++j)out.limb[j]+=packet->gradient[j];if(angular_error)*angular_error+=packet->error;continue;}}
        if(ordinary){double error=0;if(const auto pair=ordinary_arc(*ordinary,value_gate,work,error)){if(angular_error)*angular_error+=error;out.fh+=pair->flux;for(int j=0;j<8;++j)out.limb[j]+=pair->gradient[j];continue;}}
        std::array<double,8> middle_motion{},half_motion{};
        for(int j=0;j<8;++j) {
            const double dl=lo.motion[j]+lo.radial_motion*motion[j];
            const double dh=hi.motion[j]+hi.radial_motion*motion[j];
            middle_motion[j]=.5*(dl+dh);half_motion[j]=.5*(dh-dl);
        }
        struct AngularPoint {double flux;std::array<double,8> tangent;Complex unit;};
        const Complex central_unit(std::cos(mid),std::sin(mid));
        const int maximum=std::max(level,8);packet_cache_detail::DyadicPacketCache<AngularPoint> cache;
        double previous=0;bool accepted=false;
        local_precision::Arc<true> precise;
        for(int current_level=2;current_level<=maximum;++current_level) {
        const auto& rule=angular_rule_extended(current_level);const int m=1<<current_level;cache.prepare(current_level);
        double total_flux=0;std::array<double,8> total_gradient{};
        for(int k=1;k<m;++k) {
            const double x=rule.x[k-1],shape=full?1:(1-x)*(1+x),theta=mid+half*x;
            const unsigned slot=packet_cache_detail::slot(current_level,k);auto& point=cache[slot];
            if(!cache.contains(slot)) {
            double phi_error=0;Evaluation<true> jet;
            if(x==0&&have_midpoint){ray.add_parameters(jet,midpoint_core);phi_error=midpoint_core.roundoff;}
            else {
                // Nested nodes are mirrored. One sine/cosine pair supplies
                // both physical rays; keep the other unit with its packet.
                // Node/reflection rounding remains inside the physical field's
                // existing arithmetic bound. Promoted samples keep their own
                // exact angular map and do not use this approximation.
                if(k<m/2) {
                    const double d=half*x,co=std::cos(d),si=std::sin(d);
                    point.unit=central_unit*Complex(co,si);
                    cache[packet_cache_detail::slot(current_level,m-k)].unit=central_unit*Complex(co,-si);
                } else if(k==m/2)point.unit=central_unit;
                jet=ray.sample_unit(point.unit,theta,phi_error);
            }
            ++work.evaluations;
            auto& tangent=point.tangent;auto& flux=point.flux;
            if(!(jet.phi>0) || phi_error>tangent_phi_roundoff*jet.phi) {
                using Q=local_precision::Q;
                if(!precise.ready) {
                    if(full){precise.lo=Q(lo.theta);precise.hi=Q(hi.theta);}
                    else {
                        if(!local_precision::endpoint(R,lo.theta,p,precise.lo) ||
                           !local_precision::endpoint(R,hi.theta,p,precise.hi))return false;
                        const auto lq=local_precision::evaluate<true>(Q(R),precise.lo,p),
                                   hq=local_precision::evaluate<true>(Q(R),precise.hi,p);
                        for(int j=0;j<8;++j){precise.dlo[j]=-(lq.dp[j]+lq.dR*Q(motion[j]))/lq.dtheta;
                            precise.dhi[j]=-(hq.dp[j]+hq.dR*Q(motion[j]))/hq.dtheta;}
                    }
                    precise.ready=true;
                }
                const Q halfq=(precise.hi-precise.lo)/2,xq=Q(x);
                const auto q=local_precision::evaluate<true>(Q(R),precise.lo+halfq*(1+xq),p);++work.quad_samples;
                if(!(q.phi>0) || !finiteq(q.phi))return false;
                const Q root=sqrtq(q.phi/Q(shape));flux=double(Q(R)*halfq*root);
                for(int j=0;j<8;++j) {
                    const Q angle_motion=(precise.dlo[j]*(1-xq)+precise.dhi[j]*(1+xq))/2;
                    tangent[j]=double(Q(motion[j])*halfq*root+Q(R)*(halfq*(q.dp[j]+q.dR*Q(motion[j])+q.dtheta*angle_motion)/(2*root*Q(shape))+
                        root*(precise.dhi[j]-precise.dlo[j])/2));
                }
            } else {
                const double root=std::sqrt(jet.phi/shape);flux=R*half*root;
                // Contract scalar geometry factors once, then transport all
                // eight RHS. Endpoint motion is affine in the node coordinate.
                const double A=R*half/(2*root*shape),B=half*root+A*jet.dR;
                const double C=A*jet.dtheta,D=R*root+C*x;
                for(int j=0;j<8;++j)
                    tangent[j]=A*jet.dp[j]+B*motion[j]+C*middle_motion[j]+D*half_motion[j];
            }

            }
            // Differentiate the regularized endpoint map. Value and tangent
            // use the same second-kind nodes; only the tangent replay carries
            // endpoint/radial motion and its conditional precision repair.
            const double weight=full?rule.w[k-1]:rule.norm_w[k-1];
            total_flux+=weight*point.flux;
            for(int j=0;j<8;++j)total_gradient[j]+=weight*point.tangent[j];
        }
        cache.finish(current_level);
        if(short_rule&&current_level==2){if(angular_error)*angular_error+=short_bound*std::abs(total_flux)/(1-short_bound);out.fh+=total_flux;for(int j=0;j<8;++j)out.limb[j]+=total_gradient[j];accepted=true;break;}
        if(current_level>=3 && (std::abs(total_flux-previous)<=value_gate*std::abs(total_flux) || current_level==maximum)) {
            if(angular_error)*angular_error+=std::abs(total_flux-previous);out.fh+=total_flux;for(int j=0;j<8;++j)out.limb[j]+=total_gradient[j];accepted=true;break;
        }
        previous=total_flux;
        }
        if(!accepted)return false;
    }
    next_anchor.geometry_verified=true;if(anchor)*anchor=next_anchor;
    return true;
}
} // namespace shape_gradient_detail
} // namespace lcbinint::algebraic_boundary::triple
