#pragma once
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
namespace lcbinint::magnification::detail {
// Model moment for sqrt(t) on a unit lattice starting at t=delta. Include
// interior midpoint error, not just the first boundary cell. The far cutoff
// is removed with the midpoint Euler--Maclaurin tail (no new lens samples).
// This is a local asymptotic model, not a bound on arbitrary lens geometry.
inline long double power_lattice_endpoint_constant(long double delta,int degree=0) {
    constexpr int count=32;
    const long double end=delta+count-.5L;
    const auto power=[degree](long double t){long double v=std::sqrt(t);for(int j=0;j<degree;++j)v*=t;return v;};
    const long double alpha=.5L+degree,inv=1/end,p=power(end);
    long double residual=end*p/(alpha+1);
    for(int k=0;k<count;++k)residual-=power(delta+k);
    residual-=alpha*p*inv/24;
    residual+=7*alpha*(alpha-1)*(alpha-2)*p*inv*inv*inv/5760;
    residual-=31*alpha*(alpha-1)*(alpha-2)*(alpha-3)*(alpha-4)*p*inv*inv*inv*inv*inv/967680;
    return residual;
}
inline const auto& smooth_endpoint_moment_table() {
    constexpr std::size_t bins=1024;
    static const auto smooth=[] {
        std::array<std::array<double,3>,bins+1> result{};
        for(std::size_t i=0;i<=bins;++i) {
            const long double d=static_cast<long double>(i)/bins;
            for(int j=0;j<3;++j) {
                long double p=std::sqrt(d);for(int k=0;k<j;++k)p*=d;
                result[i][j]=static_cast<double>(power_lattice_endpoint_constant(d,j)+p);
            }
        }
        return result;
    }();
    return smooth;
}
inline std::array<double,3> power_endpoint_quadrature_defects(double delta,double first_weight,double second_weight=1) {
    if(!(std::isfinite(delta)&&delta>=0&&delta<=1.5&&std::isfinite(first_weight)&&std::isfinite(second_weight))) {
        const double nan=std::numeric_limits<double>::quiet_NaN();return {nan,nan,nan};
    }
    // Shifting the first retained row by one lattice site omits exactly
    // one known model sample. The recurrence is exact for each power.
    if(delta>1) {
        auto result=power_endpoint_quadrature_defects(delta-1,1,1);
        double missing=std::sqrt(delta-1),first=std::sqrt(delta),second=std::sqrt(delta+1);
        for(int degree=0;degree<3;++degree) {
            result[degree]+=missing-(first_weight-1)*first-(second_weight-1)*second;
            missing*=delta-1;first*=delta;second*=delta+1;
        }
        return result;
    }
    constexpr std::size_t bins=1024;
    const auto& smooth=smooth_endpoint_moment_table();
    const double u=delta*bins;const auto i=std::min<std::size_t>(static_cast<std::size_t>(u),bins-1);
    const auto base=std::min<std::size_t>(i?i-1:0,bins-3);const double t=u-base;
    const double w0=-(t-1)*(t-2)*(t-3)/6,w1=t*(t-2)*(t-3)/2,w2=-t*(t-1)*(t-3)/2,w3=t*(t-1)*(t-2)/6;
    double p0=std::sqrt(delta),p1=std::sqrt(delta+1);std::array<double,3> result;
    for(int degree=0;degree<3;++degree) {
        const double moment=w0*smooth[base][degree]+w1*smooth[base+1][degree]+w2*smooth[base+2][degree]+w3*smooth[base+3][degree];
        result[degree]=moment-first_weight*p0-(second_weight-1)*p1;
        p0*=delta;p1*=delta+1;
    }
    return result;
}
inline double power_endpoint_quadrature_defect(double delta,double first_weight,double second_weight=1,int degree=0) {
    if(degree==3) {
        if(!(std::isfinite(delta)&&delta>=0&&delta<=1.5&&std::isfinite(first_weight)&&std::isfinite(second_weight)))return std::numeric_limits<double>::quiet_NaN();
        if(delta>1)return power_endpoint_quadrature_defect(delta-1,1,1,3)+std::pow(delta-1,3.5)-(first_weight-1)*std::pow(delta,3.5)-(second_weight-1)*std::pow(delta+1,3.5);
        constexpr std::size_t bins=1024;
        static const auto smooth=[] {
            std::array<double,bins+1> result{};
            for(std::size_t j=0;j<=bins;++j) {
                const long double d=static_cast<long double>(j)/bins;
                result[j]=double(power_lattice_endpoint_constant(d,3)+std::pow(d,3.5L));
            }
            return result;
        }();
        const double u=delta*bins;const auto i=std::min<std::size_t>(std::size_t(u),bins-1);
        const auto base=std::min<std::size_t>(i?i-1:0,bins-3);const double t=u-base;
        const double w0=-(t-1)*(t-2)*(t-3)/6,w1=t*(t-2)*(t-3)/2,w2=-t*(t-1)*(t-3)/2,w3=t*(t-1)*(t-2)/6;
        return w0*smooth[base]+w1*smooth[base+1]+w2*smooth[base+2]+w3*smooth[base+3]-first_weight*std::pow(delta,3.5)-(second_weight-1)*std::pow(delta+1,3.5);
    }
    if(degree<0||degree>2)return std::numeric_limits<double>::quiet_NaN();
    return power_endpoint_quadrature_defects(delta,first_weight,second_weight)[degree];
}
// Regular cap brightness after integrating linear LD along a row has
// integer powers. Bernoulli endpoint moments are exact for t and t^2.
inline double integer_endpoint_quadrature_defect(double delta,double first_weight,double second_weight,int power) {
    if(!(std::isfinite(delta)&&delta>=0&&delta<=1.5&&std::isfinite(first_weight)&&std::isfinite(second_weight)&&(power>=1&&power<=4)))return std::numeric_limits<double>::quiet_NaN();
    const double d=delta;
    const double base=power==1?(d*d-d+1./6)/2:power==2?(d*d*d-1.5*d*d+.5*d)/3:
        power==3?(d*d*d*d-2*d*d*d+d*d-1./30)/4:
        (d*d*d*d*d-2.5*d*d*d*d+(5./3)*d*d*d-d/6)/5;
    return base-(first_weight-1)*std::pow(d,power)-(second_weight-1)*std::pow(d+1,power);
}
// Integrable coarea singularity at a regular image tip. Only used for
// the small LD-table bias estimate; no ray or root queries.
inline double inverse_sqrt_endpoint_quadrature_defect(double delta) {
    if(!(std::isfinite(delta)&&delta>0&&delta<=1))return std::numeric_limits<double>::quiet_NaN();
    constexpr int count=32;const long double d=delta,end=d+count-.5L;
    long double result=2*std::sqrt(end);
    for(int k=0;k<count;++k)result-=1/std::sqrt(d+k);
    result+=1/(48*end*std::sqrt(end));
    result-=7*1.875L/(5760*end*end*end*std::sqrt(end));
    return static_cast<double>(result);
}
// Nearest-bin lookup of sqrt(1-r^2) uses a trapezoidal profile integral.
// Its regular-limb leading bias is a universal h^(3/2) endpoint moment.
inline double linear_ld_table_bias_moment(int bins) {
    if(bins<=0)return std::numeric_limits<double>::quiet_NaN();
    static const double endpoint=static_cast<double>(power_lattice_endpoint_constant(0));
    const double step=1./bins;
    return endpoint*step*std::sqrt(step)-step*step/24;
}
inline double sqrt_endpoint_quadrature_defect(double delta,double first_weight,double second_weight=1) {
    return power_endpoint_quadrature_defect(delta,first_weight,second_weight,0);
}
} // namespace lcbinint::magnification::detail
