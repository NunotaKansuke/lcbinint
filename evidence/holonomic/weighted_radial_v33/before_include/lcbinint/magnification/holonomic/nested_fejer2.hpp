#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace lcbinint::holonomic {
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
inline const Fejer2Rule& fejer_rule(int level) {
    if(level<1 || level>8) throw std::invalid_argument("Fejer level must be 1..8");
    static const std::array<Fejer2Rule,8> rules={Fejer2Rule(1),Fejer2Rule(2),Fejer2Rule(3),Fejer2Rule(4),Fejer2Rule(5),Fejer2Rule(6),Fejer2Rule(7),Fejer2Rule(8)};
    return rules[level-1];
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
