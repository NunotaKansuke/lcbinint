#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace lcbinint::algebraic_boundary {
// Canonical dyadic-angle nodes: inherited coordinates are bitwise identical.
inline double fejer_node(int level,int k) {
    while (!(k&1)) { k/=2; --level; }
    return level==1 ? 0.0 : std::cos(3.14159265358979323846*k/(1<<level));
}
struct Fejer2Rule {
    std::vector<double> x,w,norm_w,interp;
    explicit Fejer2Rule(int level) {
        const int m=1<<level,n=m-1,nc=m/2-1;
        const double pi=3.14159265358979323846;
        x.resize(n); w.resize(n); norm_w.resize(n);
        interp.resize((m/2)*nc);
        for(int k=1;k<m;++k) {
            x[k-1]=fejer_node(level,k);
            double th=pi*k/m,s=std::sin(th),sum=0;
            for(int j=1;j<m;j+=2) sum+=std::sin(j*th)/j;
            w[k-1]=4*s*sum/m; norm_w[k-1]=pi*s*s/m;
            if((k&1) && nc) {
                long double den=0;
                for(int j=1;j<=nc;++j) {
                    double sj=std::sin(pi*j/(m/2));
                    double a=(j&1 ? -1 : 1)*sj*sj/(x[k-1]-fejer_node(level-1,j));
                    interp[(k/2)*nc+j-1]=a; den+=a;
                }
                for(int j=0;j<nc;++j) interp[(k/2)*nc+j]/=den;
            }
        }
    }
};
template<int Level> inline const Fejer2Rule& fejer_rule_storage() {
    static const Fejer2Rule rule(Level);return rule;
}
inline const Fejer2Rule& fejer_rule(int level) {
    // Low-order evaluations do not construct unused high-order tables.
    switch(level) {
        case 1:return fejer_rule_storage<1>();
        case 2:return fejer_rule_storage<2>();
        case 3:return fejer_rule_storage<3>();
        case 4:return fejer_rule_storage<4>();
        case 5:return fejer_rule_storage<5>();
        case 6:return fejer_rule_storage<6>();
        case 7:return fejer_rule_storage<7>();
        case 8:return fejer_rule_storage<8>();
        default:throw std::invalid_argument("Fejer level must be 1..8");
    }
}
// Gauss-Chebyshev-II uses the same nested dyadic-angle nodes.
// Integrate sqrt(1-x*x)*g(x); values supplied to the rule are g(x).
template<int Level> inline const Fejer2Rule& radial_rule_storage() {
    static const auto rule=[] {auto out=fejer_rule_storage<Level>();out.w=out.norm_w;return out;}();
    return rule;
}
inline const Fejer2Rule& radial_rule(int level,bool weighted=false) {
    if(!weighted)return fejer_rule(level);
    switch(level) {
        case 1:return radial_rule_storage<1>();
        case 2:return radial_rule_storage<2>();
        case 3:return radial_rule_storage<3>();
        case 4:return radial_rule_storage<4>();
        case 5:return radial_rule_storage<5>();
        case 6:return radial_rule_storage<6>();
        case 7:return radial_rule_storage<7>();
        case 8:return radial_rule_storage<8>();
        default:throw std::invalid_argument("Radial level must be 1..8");
    }
}
// Interpolatory product rule for (1-x*x)*g(x) on the same nested nodes.
// U_(k-1) moment = 4/[k*(4-k*k)] for odd k, zero for even k.
template<int Level> inline const Fejer2Rule& linear_radial_rule_storage() {
    static const auto rule=[] {
        Fejer2Rule out(Level);const int m=1<<Level;
        for(int i=1;i<m;++i) {
            const long double theta=3.1415926535897932384626433832795029L*i/m;
            long double sum=0;
            for(int k=1;k<m;k+=2)sum+=4*sinl(k*theta)/(k*(4.L-k*k));
            out.w[i-1]=double(2*sinl(theta)*sum/m);
        }
        return out;
    }();
    return rule;
}
inline const Fejer2Rule& linear_radial_rule(int level) {
    // Construct only a requested level. Low-order point evaluations should
    // not pay the O(n^2) setup of every unused high-order product rule.
    switch(level) {
        case 1:return linear_radial_rule_storage<1>();
        case 2:return linear_radial_rule_storage<2>();
        case 3:return linear_radial_rule_storage<3>();
        case 4:return linear_radial_rule_storage<4>();
        case 5:return linear_radial_rule_storage<5>();
        case 6:return linear_radial_rule_storage<6>();
        case 7:return linear_radial_rule_storage<7>();
        case 8:return linear_radial_rule_storage<8>();
        default:throw std::invalid_argument("Radial level must be 1..8");
    }
}
// Fold-map trigonometry depends only on canonical dyadic abscissae.
// Keep double and extended evaluations separate to preserve the incumbent
// arithmetic paths; cache sine values, not geometry-dependent products.
struct FoldMapFactors {
    double half_sine=0,sine=0;
    long double half_sine_extended=0,sine_extended=0;
};
inline const std::array<FoldMapFactors,256>& fold_map_factors() {
    static const auto factors=[] {
        std::array<FoldMapFactors,256> out{};
        for(int slot=1;slot<256;++slot) {
            const double x=fejer_node(8,slot),t=(1+x)*.5;
            const double pi=3.14159265358979323846;
            out[slot].half_sine=std::sin(pi*.5*(t<=.5?t:1-t));
            out[slot].sine=std::sin(pi*t);
            const long double tl=(1.0L+x)*.5L;
            constexpr long double pil=3.1415926535897932384626433832795029L;
            out[slot].half_sine_extended=std::sin(pil*.5L*(tl<=.5L?tl:1-tl));
            out[slot].sine_extended=std::sin(pil*tl);
        }
        return out;
    }();
    return factors;
}
// map choice is immutable throughout p-refinement. h children retain the
// parent map and partition its x domain, avoiding moving existing samples.
struct FoldRadialMap {
    double a,b;
    bool left=false,right=false;
    std::array<double,2> operator()(double x) const {
        const double t=(1+x)*0.5,w=b-a,pi=3.14159265358979323846;
        if(left && right) {
            // Form the offset at the nearest end to avoid cancellation.
            double s=std::sin(pi*0.5*(t<=0.5?t:1-t));
            return {t<=0.5 ? a+w*s*s : b-w*s*s,pi*w*0.25*std::sin(pi*t)};
        }
        if(left) return {a+w*t*t,w*t};
        if(right) return {b-w*(1-t)*(1-t),w*(1-t)};
        return {a+w*t,w*0.5};
    }
};
}
