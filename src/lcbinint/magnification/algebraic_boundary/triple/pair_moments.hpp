#pragma once
#include "root_pair.hpp"
namespace lcbinint::algebraic_boundary::triple::root_pair_detail {
template<int N> struct CofactorSeries {
    std::array<Jet<N>,17> c{};int degree=0;double radius=0;
};
template<class T,std::size_t M> std::array<T,M> translate(std::array<T,M> p,T center) {
    std::array<T,M> out{};int degree=0;
    for(int j=int(M)-1;j>=0;--j){for(int k=degree;k>0;--k)out[k]=out[k]*center+out[k-1];out[0]=out[0]*center+p[j];if(degree<int(M)-1)++degree;}
    return out;
}
template<int N> bool prepare_series(const Arc<N>& model,double value_gate,CofactorSeries<N>& result) {
    using J=Jet<N>;using I=interval_detail::Interval<double>;
    const auto& m=model.pair.mean;
    if(!(model.half_width>0 && model.half_width<.0625 && value_gate>0))return false;
    const auto q=translate(model.pair.quotient,m);
    const std::array<J,3> s{J(1)+m*m,J(2)*m,J(1)};
    std::array<std::array<J,3>,2> poles{};
    for(int j=0;j<2;++j)poles[j]=translate(model.poles[j],m);
    if(!(q[0].value>0 && poles[0][0].value>0 && poles[1][0].value>0))return false;
    struct Bound {std::array<double,5> tail{};double constant=0;int degree=0;};
    const auto bound=[&](const auto& polynomial) {
        constexpr int size=int(std::tuple_size<std::decay_t<decltype(polynomial)>>::value);
        std::array<I,size> values{};for(int j=0;j<size;++j)values[j]=I(polynomial[j].value);
        const auto shifted=translate(values,I(m.value));Bound out;out.degree=size-1;
        if(shifted[0].lo<=0 && shifted[0].hi>=0)return out;
        out.constant=std::min(std::abs(shifted[0].lo),std::abs(shifted[0].hi));
        for(int j=1;j<size;++j)out.tail[j]=std::max(std::abs(shifted[j].lo),std::abs(shifted[j].hi));return out;
    };
    const std::array<J,3> one{J(1),J(0),J(1)};
    const std::array<Bound,4> bounds{bound(model.pair.quotient),bound(one),bound(model.poles[0]),bound(model.poles[1])};
    double radius=.25;
    for(int iteration=0;;++iteration) {
        bool okay=true;
        for(const auto& b:bounds) {
            if(!(b.constant>64*std::numeric_limits<double>::min()))return false;
            double tail=b.tail[b.degree];for(int j=b.degree-1;j>=1;--j)tail=tail*(4*radius)+b.tail[j];
            tail*=4*radius;tail*=1+128*std::numeric_limits<double>::epsilon();
            okay=okay && std::isfinite(tail) && tail<.5*b.constant;
        }
        if(okay)break;
        radius*=.5;if(iteration==39 || !(radius>0))return false;
    }
    // On the certified disk every factor differs from its constant by <=1/2.
    // M_Cauchy/K_min =27, hence a radius ratio of 1/4 gives the relative
    // value tail 36*4^(-degree-1). No derivative component chooses the order.
    int degree=0;double tail=9;
    while(degree<16 && (degree<4 || tail>value_gate)){++degree;tail*=.25;}
    if(tail>value_gate)radius*=4*std::pow(value_gate/36,1./(degree+1));
    if(!(radius>4*model.half_width))return false;
    result.degree=degree;result.radius=radius;
    using Polynomial=std::array<J,17>;
    const auto product=[&](const Polynomial& a,int na,const Polynomial& b,int nb) {
        Polynomial out{};for(int i=0;i<=na;++i)for(int j=0;j<=nb && i+j<=degree;++j)out[i+j]=out[i+j]+a[i]*b[j];return out;
    };
    Polynomial S{},P1{},P2{},Q{};double power=1;
    for(int j=0;j<=4;++j){Q[j]=q[j]*J(power);if(j<3){S[j]=s[j]*J(power);P1[j]=poles[0][j]*J(power);P2[j]=poles[1][j]*J(power);}power*=radius;}
    auto denominator=product(S,2,S,2);denominator=product(denominator,4,S,2);
    denominator=product(denominator,std::min(6,degree),P1,2);denominator=product(denominator,std::min(8,degree),P2,2);
    if(!(denominator[0].value>0))return false;
    Polynomial ratio{},square{};const J inv=inverse(denominator[0]);ratio[0]=Q[0]*inv;
    if(!(ratio[0].value>0))return false;
    square[0]=root(ratio[0]);const J inv_twice_root=inverse(J(2)*square[0]),prefactor=J(2)*inverse(model.rho);
    result.c[0]=prefactor*square[0];
    for(int n=1;n<=degree;++n) {
        J sum;for(int j=1;j<=std::min(n,10);++j)sum=sum+denominator[j]*ratio[n-j];
        ratio[n]=(Q[n]-sum)*inv;sum=J(0);
        for(int j=1;j<n;++j)sum=sum+square[j]*square[n-j];
        square[n]=(ratio[n]-sum)*inv_twice_root;result.c[n]=prefactor*square[n];
    }
    for(int n=0;n<=degree;++n){if(!std::isfinite(result.c[n].value))return false;for(double p:result.c[n].dp)if(!std::isfinite(p))return false;}
    return true;
}
// Definite moments are evaluated at the current, frozen chart boundaries.
// Internal moving-cut terms cancel with the adjacent numerical tails.
template<int N> bool integrate_series(const Arc<N>& model,const CofactorSeries<N>& series,
        double ta,double tb,Jet<N>& out) {
    const double r=series.radius,xa=(ta-model.pair.mean.value)/r,xb=(tb-model.pair.mean.value)/r;
    const double e=model.pair.square.value/(r*r),h=model.half_width/r;
    if(!(xb>xa) || std::max(std::abs(xa),std::abs(xb))>1+64*std::numeric_limits<double>::epsilon() || xb-xa<1e-7)return false;
    if(e>0 && (xa*xb<0 || std::min(std::abs(xa),std::abs(xb))<h*(1-64*std::numeric_limits<double>::epsilon())))return false;
    const double sa=std::sqrt(std::max(0.,xa*xa-e)),sb=std::sqrt(std::max(0.,xb*xb-e));
    std::array<double,19> J{};J[1]=sb-sa;
    if(e<0)J[0]=std::asinh(xb/h)-std::asinh(xa/h);
    else J[0]=std::copysign(std::acosh(std::max(1.,std::abs(xb)/h)),xb)-std::copysign(std::acosh(std::max(1.,std::abs(xa)/h)),xa);
    double pa=xa,pb=xb;
    for(int n=2;n<=series.degree+2;++n){J[n]=(pb*sb-pa*sa+(n-1)*e*J[n-2])/n;pa*=xa;pb*=xb;}
    Jet<N> result;double center=0,square=0;
    for(int n=0;n<=series.degree;++n) {
        const double I=r*r*(J[n+2]-e*J[n]);const auto& c=series.c[n];
        result.value+=c.value*I;square+=c.value*J[n];
        center+=c.value*(r*J[n+1]+(n?double(n)*r*(J[n+1]-e*J[n-1]):0));
        for(int k=0;k<N;++k)result.dp[k]+=c.dp[k]*I;
    }
    for(int k=0;k<N;++k)result.dp[k]-=model.pair.mean.dp[k]*center+.5*model.pair.square.dp[k]*square;
    if(!(result.value>0) || !std::isfinite(result.value))return false;
    for(double d:result.dp)if(!std::isfinite(d))return false;
    out=result;return true;
}
} // namespace lcbinint::algebraic_boundary::triple::root_pair_detail
