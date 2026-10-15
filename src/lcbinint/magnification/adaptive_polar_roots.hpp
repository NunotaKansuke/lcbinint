#pragma once
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace lcbinint::magnification::detail {

// Radial source-limb equations become a real sextic after multiplying
// by the two lens denominators. Its derivative isolates every radial branch;
// the physical lens residual, rather than the expanded polynomial, polishes
// the actual boundaries. Long double protects nearly paired limb roots.
// Degrees never exceed six. Keep recursion and its ordered cuts on the
// stack: a column used to allocate a vector at every derivative degree.
// Extra capacity also accommodates repeated stationary roots without
// changing the original ordering or numerical operations.
struct PolarRealRoots {
    std::array<long double, 16> values{};
    std::size_t count = 0;
    bool empty() const { return count == 0; }
    std::size_t size() const { return count; }
    long double back() const { return values[count - 1]; }
    long double operator[](std::size_t i) const { return values[i]; }
    const long double* begin() const { return values.data(); }
    const long double* end() const { return values.data() + count; }
    void push_back(long double value) {
        if (count == values.size())
            throw std::runtime_error("polar polynomial root capacity exceeded");
        values[count++] = value;
    }
};

inline void polar_real_polynomial_roots(const std::array<long double, 7>& c, int degree,
    long double lo, long double hi, PolarRealRoots& roots)
{
    while (degree > 0 && c[degree] == 0.0L) --degree;
    if (degree == 0) return;
    if (degree == 1) {
        const long double r = -c[0] / c[1];
        if (r > lo && r < hi) roots.push_back(r);
        return;
    }
    if (degree == 2) {
        const long double discriminant = c[1] * c[1] - 4.0L * c[2] * c[0];
        if (discriminant < 0.0L) return;
        const long double q = -0.5L * (c[1] + std::copysign(std::sqrt(discriminant), c[1]));
        const long double first = q / c[2], second = q == 0.0L ? first : c[0] / q;
        for (long double r : {std::min(first, second), std::max(first, second)})
            if (r > lo && r < hi && (roots.empty() || r != roots.back())) roots.push_back(r);
        return;
    }
    std::array<long double, 7> derivative{};
    for (int i = 1; i <= degree; ++i) derivative[i - 1] = i * c[i];
    PolarRealRoots cuts;
    cuts.push_back(lo);
    polar_real_polynomial_roots(derivative, degree - 1, lo, hi, cuts);
    cuts.push_back(hi);
    const auto evaluate = [&](long double x) {
        long double value = c[degree];
        for (int i = degree - 1; i >= 0; --i) value = value * x + c[i];
        return value;
    };
    for (std::size_t i = 1; i < cuts.size(); ++i) {
        long double a = cuts[i - 1], b = cuts[i];
        long double fa = evaluate(a), fb = evaluate(b);
        if (fa == 0.0L && a > lo) roots.push_back(a);
        if ((fa < 0.0L) == (fb < 0.0L)) continue;
        long double x = 0.5L * (a + b);
        for (int it = 0; it < 64; ++it) {
            long double value = c[degree], slope = 0.0L;
            for (int j = degree - 1; j >= 0; --j) {
                slope = slope * x + value;
                value = value * x + c[j];
            }
            if (value == 0.0L) break;
            if ((value < 0.0L) == (fa < 0.0L)) { a = x; fa = value; }
            else { b = x; fb = value; }
            long double next = x - value / slope;
            if (!(next > a && next < b)) next = 0.5L * (a + b);
            if (next == x || std::abs(next - x) <= 4.0L * std::numeric_limits<long double>::epsilon() * std::max(1.0L, std::abs(x))) {
                x = next; break;
            }
            x = next;
        }
        roots.push_back(x);
    }
}

// Continuation proposes a partition; Bernstein sign variations check its
// completeness over the entire radial domain. Zero variations exclude roots,
// one variation plus opposite endpoint signs isolates exactly one root.
// Ambiguous signs, multiple roots and branch changes use the original solver.
// The roundoff guard is deliberately conservative, not a formal interval proof.
template<typename T>
inline int polar_stationary_interval_variations(const std::array<long double,7>& wide,
    long double wide_a,long double wide_b)
{
    static constexpr int choose[6][6]={{1},{1,1},{1,2,1},{1,3,3,1},{1,4,6,4,1},{1,5,10,10,5,1}};
    static constexpr T weights[6][6]={{1},{1,T(1)/5},{1,T(2)/5,T(1)/10},
        {1,T(3)/5,T(3)/10,T(1)/10},{1,T(4)/5,T(6)/10,T(4)/10,T(1)/5},{1,1,1,1,1,1}};
    const T a=static_cast<T>(wide_a),width=static_cast<T>(wide_b)-a;
    if(!(width>0)) return -1;
    std::array<T,6> c{},ap{},wp{},power{},magnitude{};ap[0]=wp[0]=1;
    for(int j=0;j<6;++j){
        c[j]=static_cast<T>(wide[j]);
        if(!std::isfinite(c[j]) || (wide[j]!=0 && std::abs(c[j])<std::numeric_limits<T>::min())) return -1;
    }
    for(int j=1;j<6;++j){ap[j]=ap[j-1]*a;wp[j]=wp[j-1]*width;}
    for(int j=0;j<6;++j)for(int i=j;i<6;++i){
        const T term=c[i]*choose[i][j]*ap[i-j]*wp[j];
        power[j]+=term;magnitude[j]+=std::abs(term);
    }
    int last=0,variations=0;
    for(int k=0;k<6;++k){
        T value=0,scale=0;
        for(int j=0;j<=k;++j){value+=power[j]*weights[k][j];scale+=magnitude[j]*weights[k][j];}
        if(scale==0)continue;
        if(!std::isfinite(value) || !std::isfinite(scale) ||
            std::abs(value)<=512*(std::numeric_limits<T>::epsilon()*scale+std::numeric_limits<T>::min())) return -1;
        const int sign=value<0?-1:1;
        if(last && last!=sign)++variations;
        last=sign;
    }
    return variations;
}

inline bool polar_continue_stationary_roots(const std::array<long double, 7>& c,
    long double upper, const PolarRealRoots& previous, PolarRealRoots& roots)
{
    const auto evaluate = [&](long double x) {
        long double f=c[5];
        for (int j=4;j>=0;--j) f=f*x+c[j];
        return f;
    };
    std::array<long double,18> cuts{};
    std::size_t count=1;
    for (std::size_t i=1;i<previous.size();++i)
        cuts[count++]=.5L*(previous[i-1]+previous[i]);
    cuts[count++]=upper;
    for (std::size_t interval=1;interval<count;++interval) {
        const long double a=cuts[interval-1], b=cuts[interval];
        int variations=polar_stationary_interval_variations<double>(c,a,b);
        if(variations<0) variations=polar_stationary_interval_variations<long double>(c,a,b);
        if(variations<0) return false;
        if (!variations) continue;
        if (variations!=1) return false;
        long double lo=a,hi=b,fa=evaluate(lo),fb=evaluate(hi);
        if (!std::isfinite(fa) || !std::isfinite(fb) || fa==0 || fb==0 || (fa<0)==(fb<0))
            return false;
        long double x=interval-1<previous.size() ? previous[interval-1] : .5L*(lo+hi);
        if (!(x>lo && x<hi)) x=.5L*(lo+hi);
        bool converged=false;
        for (int it=0;it<64;++it) {
            long double f=c[5],df=0;
            for (int j=4;j>=0;--j) { df=df*x+f;f=f*x+c[j]; }
            if (f==0) { converged=true;break; }
            if ((f<0)==(fa<0)) { lo=x;fa=f; } else hi=x;
            long double next=x-f/df;
            if (!(next>lo && next<hi)) next=.5L*(lo+hi);
            // These extrema become double partition cuts and are not the
            // physical limb roots. Stop once their consumed coordinates are
            // stable; the Bernstein root count and limb brackets still check
            // completeness and physical inside/outside signs.
            if (static_cast<double>(next)==static_cast<double>(x) ||
                std::abs(next-x)<=4*std::numeric_limits<long double>::epsilon()*std::max(1.0L,std::abs(x))) {
                x=next;converged=true;break;
            }
            x=next;
        }
        if (!converged) return false;
        roots.push_back(x);
    }
    return true;
}

// Check the full sextic on a proposed radial partition without solving its
// stationary quintic. Each interval must exclude roots or isolate one root.
// A double-precision sign filter has a conservative cancellation margin;
// ambiguous coefficients first use long double, then the stationary solver. The caller
// also checks physical inside/outside signs at every partition endpoint.
inline bool polar_checked_limb_partition(const std::array<long double,7>& wide,
    double upper,const PolarRealRoots& proposed,std::array<int,18>& counts,bool* used_wide=nullptr)
{
    if(used_wide)*used_wide=false;
    static constexpr int choose[7][7]={
        {1},{1,1},{1,2,1},{1,3,3,1},{1,4,6,4,1},{1,5,10,10,5,1},{1,6,15,20,15,6,1}};
    static constexpr double weights[7][7]={
        {1},{1,1.0/6},{1,2.0/6,1.0/15},{1,3.0/6,3.0/15,1.0/20},
        {1,4.0/6,6.0/15,4.0/20,1.0/15},
        {1,5.0/6,10.0/15,10.0/20,5.0/15,1.0/6},{1,1,1,1,1,1,1}};
    std::array<double,7> c{};
    for(int j=0;j<7;++j) {
        c[j]=static_cast<double>(wide[j]);
        if(!std::isfinite(c[j]) || (wide[j]!=0 && std::abs(c[j])<std::numeric_limits<double>::min())) return false;
    }
    if (!(c[0]>0) || proposed.size()>16) return false;
    std::array<double,18> cuts{};
    for(std::size_t j=0;j<proposed.size();++j) cuts[j+1]=static_cast<double>(proposed[j]);
    cuts[proposed.size()+1]=upper;
    for(std::size_t interval=1;interval<proposed.size()+2;++interval) {
        const double a=cuts[interval-1],width=cuts[interval]-a;
        if(!(width>0) || a<0 || !std::isfinite(width)) return false;
        std::array<double,7> ap{},wp{},power{},magnitude{};ap[0]=wp[0]=1;
        for(int j=1;j<7;++j) {ap[j]=ap[j-1]*a;wp[j]=wp[j-1]*width;}
        for(int j=0;j<7;++j) for(int i=j;i<7;++i) {
            const double term=c[i]*choose[i][j]*ap[i-j]*wp[j];
            power[j]+=term;magnitude[j]+=std::abs(term);
        }
        std::array<long double,7> wide_power,wide_magnitude;
        bool wide_ready=false;
        int last=0,variation=0;
        for(int k=0;k<7;++k) {
            double value=0,scale=0;
            for(int j=0;j<=k;++j) {value+=power[j]*weights[k][j];scale+=magnitude[j]*weights[k][j];}
            if(!std::isfinite(value) || !std::isfinite(scale)) return false;
            if(scale==0) continue;
            const double margin=512*(std::numeric_limits<double>::epsilon()*scale+std::numeric_limits<double>::min());
            int sign=value<0 ? -1 : 1;
            if(std::abs(value)<=margin) {
                if(used_wide)*used_wide=true;
                if(!wide_ready) {
                    // Reconstruct the consumed double cuts exactly in the
                    // wider basis, including their subtraction. Only an
                    // ambiguous coefficient pays for this transform.
                    const long double wa=cuts[interval-1];
                    const long double ww=static_cast<long double>(cuts[interval])-wa;
                    std::array<long double,7> wap,wwp;
                    wap[0]=wwp[0]=1;
                    for(int j=1;j<7;++j){wap[j]=wap[j-1]*wa;wwp[j]=wwp[j-1]*ww;}
                    wide_power.fill(0);wide_magnitude.fill(0);
                    for(int j=0;j<7;++j)for(int i=j;i<7;++i){
                        const long double term=wide[i]*choose[i][j]*wap[i-j]*wwp[j];
                        wide_power[j]+=term;wide_magnitude[j]+=std::abs(term);
                    }
                    wide_ready=true;
                }
                long double wider_value=0,wider_scale=0;
                for(int j=0;j<=k;++j){
                    const long double weight=static_cast<long double>(choose[k][j])/choose[6][j];
                    wider_value+=wide_power[j]*weight;wider_scale+=wide_magnitude[j]*weight;
                }
                const long double wider_margin=512*(std::numeric_limits<long double>::epsilon()*wider_scale+
                    std::numeric_limits<long double>::min());
                if(!std::isfinite(wider_value) || !std::isfinite(wider_scale) || std::abs(wider_value)<=wider_margin)return false;
                sign=wider_value<0 ? -1 : 1;
            }
            if(last && last!=sign) ++variation;
            last=sign;
        }
        if(variation>1) return false;
        counts[interval-1]=variation;
    }
    return true;
}

} // namespace lcbinint::magnification::detail
