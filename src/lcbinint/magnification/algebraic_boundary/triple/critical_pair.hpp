#pragma once
#include "root_pair.hpp"
#include "local_precision.hpp"
#include <type_traits>
namespace lcbinint::algebraic_boundary::triple::root_pair_detail {
template<class T> T scalar_abs(T x){if constexpr(std::is_same_v<T,__float128>)return fabsq(x);else return std::abs(x);}
template<class T> std::array<T,7> scalar_coefficients(double R,const Geometry& p) {
    using C=coefficient_detail::C<T>;
    const std::array<C,3> lenses{C{},C{T(p.a)},C{T(p.bx),T(p.by)}};
    return spectrum_coefficients(T(R),lenses,std::array<T,3>{T(1)-T(p.m1)-T(p.m2),T(p.m1),T(p.m2)},C{T(p.X),T(p.Y)},T(p.rho));
}
template<class T> std::array<T,7> precise_harmonics(double R,const Geometry& p){
 using C=coefficient_detail::C<T>;return spectrum_harmonics(T(R),std::array<C,3>{C{},C{T(p.a)},C{T(p.bx),T(p.by)}},std::array<T,3>{T(1)-T(p.m1)-T(p.m2),T(p.m1),T(p.m2)},C{T(p.X),T(p.Y)},T(p.rho));
}
template<class T> std::array<T,7> precise_harmonic_polynomial(std::array<T,7> h,double center){
 T co,si;if constexpr(std::is_same_v<T,__float128>){co=cosq(T(center));si=sinq(T(center));}else{co=std::cos(T(center));si=std::sin(T(center));}
 const T c2=co*co-si*si,s2=2*co*si;const std::array<T,3> c{co,c2,c2*co-s2*si},s{si,s2,s2*co+c2*si};for(int j=0;j<3;++j){const T re=h[2*j+1],im=h[2*j+2];h[2*j+1]=re*c[j]-im*s[j];h[2*j+2]=re*s[j]+im*c[j];}return harmonic_polynomial(h);
}
template<class T> std::array<T,7> scalar_rotate(std::array<T,7> p,double center) {
    const double pi=std::acos(-1.);const bool reciprocal=std::abs(std::remainder(center,2*pi))>pi*.5;
    const double base=reciprocal?pi:0;const auto turns=std::llround((center-base)/(2*pi));
    if(reciprocal){const auto original=p;for(int j=0;j<7;++j)p[j]=T(j%2?-1:1)*original[6-j];}
    T wide_pi,kappa;
    if constexpr(std::is_same_v<T,__float128>){wide_pi=acosq(T(-1));kappa=tanq((T(center)-T(reciprocal?1:0)*wide_pi-T(2)*T(turns)*wide_pi)/2);}
    else {wide_pi=acosl(-1.L);kappa=std::tan((T(center)-T(reciprocal?1:0)*wide_pi-T(2)*T(turns)*wide_pi)/2);}
    const T f=1+kappa*kappa,normal=T(1)/(f*f*f);
    std::array<T,7> power{},out{};power[0]=1;for(int j=1;j<7;++j)power[j]=power[j-1]*kappa;
    for(int i=0;i<7;++i)for(int j=0;j<7;++j) {
        T weight=0;for(int a=std::max(0,i-(6-j));a<=std::min(i,j);++a)
            weight+=T(binomial(j,a)*binomial(6-j,i-a))*power[j-a]*power[i-a]*T((i-a)%2?-1:1);
        out[i]+=weight*normal*p[j];
    }
    return out;
}
template<class T> bool correct_complex_pair(const std::array<T,7>& p,T& m,T& v) {
    const T initial_m=m,initial_v=v;
    T eps;if constexpr(std::is_same_v<T,__float128>)eps=ldexpq(T(1),-112);else eps=std::numeric_limits<T>::epsilon();
    for(int iteration=0;iteration<12;++iteration) {
        std::array<T,7> a{};int degree=0;
        for(int j=6;j>=0;--j){for(int k=degree;k>0;--k)a[k]=a[k]*m+a[k-1];a[0]=a[0]*m+p[j];if(degree<6)++degree;}
        const T v2=v*v,E=a[0]+v*a[2]+v2*a[4]+v2*v*a[6],O=a[1]+v*a[3]+v2*a[5];
        const T em=a[1]+3*v*a[3]+5*v2*a[5],ev=a[2]+2*v*a[4]+3*v2*a[6],om=2*ev,ov=a[3]+2*v*a[5],det=em*ov-ev*om;
        if(det==0)return false;
        const T dm=(ev*O-ov*E)/det,dv=(om*E-em*O)/det;m+=dm;v+=dv;
        if(!(v<0) || !std::isfinite(double(m)) || !std::isfinite(double(v)) ||
           double(scalar_abs(m-initial_m))>.5*std::sqrt(double(-initial_v))+1e-12*(1+std::abs(double(initial_m))) || scalar_abs(v-initial_v)>T(.75)*scalar_abs(initial_v))return false;
        if(scalar_abs(dm)<32*eps*(1+scalar_abs(m)) && scalar_abs(dv)<64*eps*(1+m*m))return true;
    }
    return false;
}
template<int N,class T> bool apply_rotated_complex_pair(Arc<N>& model,const std::array<T,7>& pc,double m0,double v0) {
    T m=T(m0),v=T(v0);
    if(!correct_complex_pair(pc,m,v))return false;
    if(!model.pair.finish(model.polynomial,double(m),double(v)))return false;
    const T c=m*m-v,two_m=2*m;std::array<T,5> q{};q[4]=pc[6];q[3]=pc[5]+two_m*q[4];
    for(int j=2;j>=0;--j)q[j]=pc[j+2]+two_m*q[j+1]-c*q[j+2];
    for(int j=0;j<5;++j)model.pair.quotient[j].value=double(q[j]);
    model.half_width=std::sqrt(-model.pair.square.value);
    return model.half_width>0;
}
template<int N,class T> bool apply_complex_pair(Arc<N>& model,const std::array<T,7>& original,double m0,double v0){return apply_rotated_complex_pair(model,scalar_rotate(original,model.center),m0,v0);}
// Promote scalar *coefficient values* locally, not the eight parameter RHS
// or every angular point. Existing primal roots/proofs and all other radii
// remain intact. The original physical minimum checks the promoted model.
template<int N,class Work> std::optional<Arc<N>> complex_model(const Arc<N>& base,
        double R,const Geometry& p,double theta,double scale,double& phi,double& phi_error,
        Coefficients<N>& coefficients,Work& work) {
    const double t0=std::tan((theta-base.center)*.5),s=scale*(1+t0*t0)*.5,v0=-s*s;
    if(!(v0<0) || !std::isfinite(v0))return std::nullopt;
    if(!(phi>0 && phi_error<1e-6*phi)) {
        const auto q=local_precision::evaluate<false>(local_precision::Q(R),local_precision::Q(theta),p);
        ++work.quad_samples;phi=double(q.phi);phi_error=64*std::numeric_limits<double>::epsilon()*std::abs(phi);
    }
    if(!(phi>0))return std::nullopt;
    const auto matches=[&](const Arc<N>& model) {
        const double delta=t0-model.pair.mean.value;
        const double predicted=model.phi_from_factor(t0,delta*delta-model.pair.square.value);
        return std::isfinite(predicted) && std::abs(predicted-phi)+phi_error<1e-6*phi;
    };
    Arc<N> model=base;
    if(model.pair.initialize_complex(model.polynomial,t0,v0)) {
        model.half_width=std::sqrt(-model.pair.square.value);
        if(matches(model)){++work.critical_pairs;return model;}
    }
    if(!coefficients.wide_harmonics){coefficients.wide_harmonics=precise_harmonics<long double>(R,p);++work.wide_coefficient_packets;}
    if(apply_rotated_complex_pair(model,precise_harmonic_polynomial(*coefficients.wide_harmonics,model.center),t0,v0)&&matches(model)){++work.critical_pairs;return model;}
    if(!coefficients.quad_harmonics){coefficients.quad_harmonics=precise_harmonics<__float128>(R,p);++work.quad_coefficient_packets;}
    if(apply_rotated_complex_pair(model,precise_harmonic_polynomial(*coefficients.quad_harmonics,model.center),t0,v0)&&matches(model)){++work.critical_pairs;return model;}
    return std::nullopt;
}
} // namespace lcbinint::algebraic_boundary::triple::root_pair_detail
