#pragma once
#include "lcbinint/magnification/holonomic/gm_physical_seed.hpp"

namespace lcbinint::holonomic {
// Forward sensitivities retain the compensated precision of the value lane.
struct GmDDDual5 {
    DD value{};
    std::array<DD,5> deriv{};
    GmDDDual5()=default;
    GmDDDual5(double x):value(x){}
    explicit GmDDDual5(DD x):value(x){}
    explicit operator double()const{return (double)value;}
    static GmDDDual5 variable(int j,double x){GmDDDual5 r(x);r.deriv[j]=DD(1);return r;}
};
inline GmDDDual5 operator+(const GmDDDual5&a,const GmDDDual5&b){GmDDDual5 r(a.value+b.value);for(int j=0;j<5;++j)r.deriv[j]=a.deriv[j]+b.deriv[j];return r;}
inline GmDDDual5 operator-(const GmDDDual5&a,const GmDDDual5&b){GmDDDual5 r(a.value-b.value);for(int j=0;j<5;++j)r.deriv[j]=a.deriv[j]-b.deriv[j];return r;}
inline GmDDDual5 operator-(const GmDDDual5&a){return GmDDDual5(0)-a;}
inline GmDDDual5 operator*(const GmDDDual5&a,const GmDDDual5&b){GmDDDual5 r(a.value*b.value);for(int j=0;j<5;++j)r.deriv[j]=a.deriv[j]*b.value+a.value*b.deriv[j];return r;}
inline GmDDDual5 operator/(const GmDDDual5&a,const GmDDDual5&b){GmDDDual5 r(a.value/b.value);for(int j=0;j<5;++j)r.deriv[j]=(a.deriv[j]-r.value*b.deriv[j])/b.value;return r;}
inline double gm_abs_value(const GmDDDual5&a){return gm_abs_value(a.value);}
inline bool gm_finite(const GmDDDual5&a){if(!gm_finite(a.value))return false;for(auto d:a.deriv)if(!gm_finite(d))return false;return true;}
inline double gm_quality_limit(const GmDDDual5&){return 1e-12;}
inline DD gm_seed_sqrt(DD x){return qsqrt_(x);}
inline GmDDDual5 gm_seed_sqrt(const GmDDDual5& x){GmDDDual5 r(qsqrt_(x.value));for(int j=0;j<5;++j)r.deriv[j]=x.deriv[j]/(DD(2)*r.value);return r;}
inline double gm_state_error(DD a,DD b,DD initial){return gm_abs_value(a-b)/(1+std::max(gm_abs_value(a),gm_abs_value(initial)));}
inline double gm_state_error(const GmDDDual5&a,const GmDDDual5&b,const GmDDDual5&initial){
    double e=gm_state_error(a.value,b.value,initial.value);
    for(int j=0;j<5;++j)e=std::max(e,gm_state_error(a.deriv[j],b.deriv[j],initial.deriv[j]));return e;
}
template<class S> GmParams<S> gm_sensitivity_params(const PrimaryFrame& pf){
    if constexpr(std::is_same_v<S,GmDDDual5>)return {S::variable(0,pf.X),S::variable(1,pf.Y),S::variable(2,pf.rho),S::variable(3,pf.m0),S::variable(4,pf.a)};
    else return {S(pf.X),S(pf.Y),S(pf.rho),S(pf.m0),S(pf.a)};
}
// Exact polynomial construction over the scalar/dual ring. This avoids
// differentiating a double-rounded polynomial when seeding sensitivities.
template<class S> std::array<S,5> gm_physical_P(double r,const GmParams<S>&p){
    using C=GmComplex<S>;S R(r),R2=R*R;C z(p.X,p.Y);
    C n0(-p.X*R2,-p.Y*R2);
    C n1=gm_complex_scale(z,p.a*R)+C(R*(R2-S(1)),S(0));
    C n2(p.a*(p.m0-R2),S(0));
    C c0=n0+n1+n2,d=n2-n0,c1(-S(2)*d.im,S(2)*d.re),c2=n1-n0-n2;
    S k=p.rho*p.rho*R2,bm=(R-p.a)*(R-p.a),bp=(R+p.a)*(R+p.a);
    return {k*bm-gm_complex_norm(c0),-S(2)*gm_complex_re_cross(c0,c1),
        k*(bm+bp)-gm_complex_norm(c1)-S(2)*gm_complex_re_cross(c0,c2),
        -S(2)*gm_complex_re_cross(c1,c2),k*bp-gm_complex_norm(c2)};
}
template<class S> std::array<S,7> gm_physical_h(double r,const GmParams<S>&p){
    auto P=gm_physical_P(r,p);S R(r),bp=(R+p.a)*(R+p.a),bm=(R-p.a)*(R-p.a);
    std::array<S,6> inner{};std::array<S,7> h{};
    for(int j=0;j<4;++j){inner[j]=inner[j]+bm*S(j+1)*P[j+1];inner[j+2]=inner[j+2]+bp*S(j+1)*P[j+1];}
    for(int j=0;j<5;++j)inner[j+1]=inner[j+1]+S(2)*bp*P[j];
    for(int j=0;j<7;++j)h[j]=((j?inner[j-1]:S(0))+(j<=4?S(2)*bp*P[j]:S(0)))/(S(8)*p.a*R);
    return h;
}
template<class S> bool gm_seed_sensitivity(double r,const GmPhysicalSeed& geometry,
    std::array<S,7>& state,int n=256){
    if(!geometry.ok||n<8)return false;
    auto p=gm_sensitivity_params<S>(geometry.chart_pf);auto P=gm_physical_P(r,p);
    auto root=[&](double initial){S t(initial);for(int iter=0;iter<5;++iter){
        S v=P[4],d(0);for(int k=3;k>=0;--k){d=d*t+v;v=v*t+P[k];}t=t-v/d;}return t;};
    S lo=root(geometry.t_lo),hi=root(geometry.t_hi);
    if(!gm_finite(lo)||!gm_finite(hi)||!((double)hi>(double)lo))return false;
    S beta=-(lo+hi),gamma=lo*hi,d1=P[3]-beta*P[4],d0=P[2]-beta*d1-gamma*P[4];
    S mid=(lo+hi)/S(2),half=(hi-lo)/S(2),R(r),bm=(R-p.a)*(R-p.a),bp=(R+p.a)*(R+p.a);
    state.fill(S(0));
    for(int j=0;j<n;++j){S t=mid+half*S(std::cos((j+0.5)*M_PI/n));
        S g=-(d0+t*(d1+t*P[4]))*(S(1)+t*t)*(bm+bp*t*t);
        if(!((double)g>0)||!gm_finite(g))return false;
        S weight=S(2*M_PI/n)/gm_seed_sqrt(g),tk(1);
        for(int k=0;k<7;++k){state[k]=state[k]+weight*tk;tk=tk*t;}}
    for(auto x:state)if(!gm_finite(x))return false;
    return true;
}
} // namespace lcbinint::holonomic
