#pragma once
#include "angular_topology.hpp"
#include "radial_polynomial.hpp"
#if defined(__SSE2__)
#include <emmintrin.h>
#endif
namespace lcbinint::algebraic_boundary::triple::surface_detail {
using I=angular_detail::I<double>;
// R-polynomial used only once to construct the bivariate boundary tensor.
inline I inverse(I x){if(!(x.lo>0))return {-INFINITY,INFINITY};
    return {angular_detail::enclosure::down(1/x.hi),angular_detail::enclosure::up(1/x.lo)};}
inline I fraction(double point,double lo,double hi) {
    const double numerator=point-lo,denominator=hi-lo;
    auto exact_difference=[](double a,double b,double value) {
        const double bvirtual=a-value,avirtual=value+bvirtual;
        return (a-avirtual)+(bvirtual-b)==0;
    };
    if(denominator>0 && exact_difference(point,lo,numerator) && exact_difference(hi,lo,denominator)) {
        if(denominator>=std::ldexp(1.,-900))for(double f:{.5,.4375})
            if(std::fma(denominator,f,-numerator)==0)return I(f);
        const double f=numerator/denominator;
        return {angular_detail::enclosure::down(f),angular_detail::enclosure::up(f)};
    }
    return (I(point)-I(lo))*inverse(I(hi)-I(lo));
}
struct Tensor {
    int nr=8,nt=6;
    std::array<std::array<I,7>,9> b{};
};
inline I ratio(int numerator,int denominator){
    if(numerator==denominator)return I(1);
    double value=double(numerator)/denominator;
    return {angular_detail::enclosure::down(value),angular_detail::enclosure::up(value)};
}
inline Tensor radial_base(double rlo,double rhi,const Geometry& p){
    RadialPolynomial R;R.degree=1;R.c[1]=I(1);
    const auto coeff=boundary_from_radius(R,p).coefficients;
    std::array<I,9> ap{},wp{};ap[0]=wp[0]=I(1);
    for(int k=1;k<9;++k){ap[k]=ap[k-1]*I(rlo);wp[k]=wp[k-1]*(I(rhi)-I(rlo));}
    Tensor local,out;
    // Separate translation/scaling from the power-to-Bernstein transform.
    // The old fourfold loop repeated the same translated coefficient once
    // for every Bernstein index and once for each projective chart.
    for(int k=0;k<9;++k)for(int h=k;h<9;++h) {
        const I weight=I(angular_detail::choose(h,k))*ap[h-k]*wp[k];
        for(int t=0;t<7;++t)local.b[k][t]=local.b[k][t]+coeff[t].c[h]*weight;
    }
    for(int i=0;i<9;++i)for(int k=0;k<=i;++k) {
        const I weight=ratio(angular_detail::choose(i,k),angular_detail::choose(8,k));
        for(int t=0;t<7;++t)out.b[i][t]=out.b[i][t]+local.b[k][t]*weight;
    }
    return out;
}
inline Tensor angular_chart(const Tensor& base,bool reciprocal){
    static const auto weights=[] {
        std::array<std::array<I,7>,7> out{};
        // C(6,k) divides 60, so the entire chart conversion weight has an
        // exact integer numerator. Enclose one division, not a cancelling sum.
        for(int j=0;j<7;++j)for(int h=0;h<7;++h) {
            int numerator=0;
            for(int k=0;k<=std::min(j,h);++k)
                numerator+=((h-k)%2?-1:1)*angular_detail::choose(h,k)*(1<<k)*
                    angular_detail::choose(j,k)*(60/angular_detail::choose(6,k));
            out[j][h]=numerator%60==0?I(numerator/60):ratio(numerator,60);
        }
        return out;
    }();
    Tensor out;
    for(int j=0;j<7;++j)for(int h=0;h<7;++h)for(int i=0;i<9;++i) {
        I c=reciprocal?base.b[i][6-h]:base.b[i][h];
        if(reciprocal && h%2)c=-c;
        out.b[i][j]=out.b[i][j]+c*weights[j][h];
    }
    return out;
}
inline Tensor make(double rlo,double rhi,bool reciprocal,const Geometry& p){
    return angular_chart(radial_base(rlo,rhi,p),reciprocal);
}
// Only called after a finite, comfortably bounded parent-hull check.
// Every subsequent convex combination remains finite. Preserve signed-zero
// and subnormal outward rounding without repeating NaN/infinity dispatch.
inline double finite_up(double value) {
    std::uint64_t bits;std::memcpy(&bits,&value,sizeof bits);
    const bool zero=(bits&UINT64_C(0x7fffffffffffffff))==0;
    bits+=UINT64_C(1)-(bits>>63)*UINT64_C(2);
    bits=zero?UINT64_C(1):bits;
    std::memcpy(&value,&bits,sizeof bits);return value;
}
#if defined(__SSE2__)
using interval_detail::packed_up;
using interval_detail::endpoints;
using interval_detail::signed_endpoints;
inline I bounded_interpolate(I a,I b,I fraction) {
    const __m128d start=signed_endpoints(a);
    const __m128d delta=packed_up(_mm_sub_pd(signed_endpoints(b),start));
    const __m128d positive=_mm_cmpge_pd(delta,_mm_setzero_pd());
    const __m128d weight=_mm_or_pd(_mm_and_pd(positive,_mm_set1_pd(fraction.hi)),
                                   _mm_andnot_pd(positive,_mm_set1_pd(fraction.lo)));
    return endpoints(packed_up(_mm_add_pd(start,packed_up(_mm_mul_pd(weight,delta)))));
}
#endif
inline I bounded_midpoint(I a,I b) {
#if defined(__SSE2__)
    return endpoints(packed_up(_mm_mul_pd(_mm_add_pd(signed_endpoints(a),signed_endpoints(b)),
                                         _mm_set1_pd(.5))));
#else
    return {-finite_up(-(a.lo+b.lo)*.5),finite_up((a.hi+b.hi)*.5)};
#endif
}
inline I interpolate(I a,I b,I fraction) {
    // For 0<=f<=1, interpolation is increasing in both endpoint values.
    // Keep f and 1-f correlated instead of multiplying independent intervals.
    const double lo=interval_detail::down(b.lo-a.lo),hi=interval_detail::up(b.hi-a.hi);
    if(fraction.lo>=0 && fraction.hi<=1 && std::isfinite(lo) && std::isfinite(hi))
        return {interval_detail::down(a.lo+interval_detail::down((lo>=0?fraction.lo:fraction.hi)*lo)),
                interval_detail::up(a.hi+interval_detail::up((hi>=0?fraction.hi:fraction.lo)*hi))};
    return (I(1)-fraction)*a+fraction*b;
}
template<int Weight> inline std::pair<Tensor,Tensor> split_weight(const Tensor& parent,int axis,I fraction){
    Tensor left=parent,right=parent;
    const int n=axis?parent.nt:parent.nr,other=axis?parent.nr:parent.nt;
    for(int k=0;k<=other;++k){
        std::array<I,9> work{};
        for(int j=0;j<=n;++j)work[j]=axis?parent.b[k][j]:parent.b[j][k];
        for(int level=0;level<=n;++level){
            if(axis){left.b[k][level]=work[0];right.b[k][n-level]=work[n-level];}
            else {left.b[level][k]=work[0];right.b[n-level][k]=work[n-level];}
            for(int j=0;j<n-level;++j) {
                const I a=work[j],b=work[j+1];
                if constexpr(Weight==0) {
                    // Scaling a rounded sum by 1/2 is exact except at the
                    // subnormal boundary; one outward neighbour also covers
                    // that possible double rounding. Avoid a second interval
                    // operation and its general sign dispatch.
                    work[j]=bounded_midpoint(a,b);
                } else if constexpr(Weight==1) {
                    // Both weights are positive exact dyadics. Endpoint
                    // multiplication needs no general interval sign dispatch.
                    work[j]={interval_detail::down(interval_detail::down(.5625*a.lo)+interval_detail::down(.4375*b.lo)),
                             interval_detail::up(interval_detail::up(.5625*a.hi)+interval_detail::up(.4375*b.hi))};
#if defined(__SSE2__)
                } else if constexpr(Weight==3) {
                    work[j]=bounded_interpolate(a,b,fraction);
#endif
                } else work[j]=interpolate(a,b,fraction);
            }
        }
    }
    return {left,right};
}
// A de Casteljau midpoint is a convex combination. Its absolute error
// grows at most linearly with depth. Bound the whole row once, then add
// that bound at the two published edges of each level.
template<int Degree,bool Angular>
inline void split_midpoint_row(const Tensor& parent,int row,I fraction,Tensor& left,Tensor& right) {
    std::array<I,Degree+1> work;
    double magnitude=0;
    for(int j=0;j<=Degree;++j) {
        work[j]=Angular?parent.b[row][j]:parent.b[j][row];
        magnitude=std::max({magnitude,std::abs(work[j].lo),std::abs(work[j].hi)});
    }
    // Each interpolation at the true fraction can differ from the exact
    // midpoint by at most 2*M*|f-1/2|. The extra epsilon factors bound
    // rounding of the sum, scaling, and accumulated error bound, including
    // gradual underflow. Callers bound M away from overflow.
    const double displacement=std::max(.5-fraction.lo,fraction.hi-.5);
    const double step=4*magnitude*(std::numeric_limits<double>::epsilon()+displacement)
        +4*std::numeric_limits<double>::denorm_min();
    for(int level=0;level<=Degree;++level) {
        auto edge=[&](I a) {
            if(level==0)return a;
            const double pad=level*step;
            return I{-finite_up(-(a.lo-pad)),finite_up(a.hi+pad)};
        };
        (Angular?left.b[row][level]:left.b[level][row])=edge(work[0]);
        (Angular?right.b[row][Degree-level]:right.b[Degree-level][row])=edge(work[Degree-level]);
        for(int j=0;j<Degree-level;++j) {
            work[j].lo=(work[j].lo+work[j+1].lo)*.5;
            work[j].hi=(work[j].hi+work[j+1].hi)*.5;
        }
    }
}
inline std::pair<Tensor,Tensor> split(const Tensor& parent,int axis,I fraction){
    if(fraction.lo==.4375 && fraction.hi==.4375)return split_weight<1>(parent,axis,fraction);
    if(fraction.lo>=0 && fraction.lo<=fraction.hi && fraction.hi<=1) {
        // Convex combinations stay within the parent hull up to rounding.
        // Leave room for endpoint differences, products and additions in all
        // eight levels before using the finite-only outward rounding kernel.
        constexpr double limit=std::numeric_limits<double>::max()/16;
        bool bounded=true;
        for(int i=0;i<=parent.nr;++i)for(int j=0;j<=parent.nt;++j)
            bounded=bounded && parent.b[i][j].lo>-limit && parent.b[i][j].hi<limit &&
                parent.b[i][j].lo<=parent.b[i][j].hi;
        if(bounded) {
            if(parent.nr==8 && parent.nt==6 && fraction.lo>=.5-1e-10 &&
               fraction.hi<=.5+1e-10) {
                Tensor left,right;
                if(axis)for(int i=0;i<=8;++i)split_midpoint_row<6,true>(parent,i,fraction,left,right);
                else for(int j=0;j<=6;++j)split_midpoint_row<8,false>(parent,j,fraction,left,right);
                return {left,right};
            }
            if(fraction.lo==.5 && fraction.hi==.5)return split_weight<0>(parent,axis,fraction);
#if defined(__SSE2__)
            return split_weight<3>(parent,axis,fraction);
#endif
        }
    }
    return split_weight<2>(parent,axis,fraction);
}
// Broad boxes need only exclusion signs, not extrema. Stop as soon as
// two coefficients forbid a common strict sign. Comparisons of represented
// endpoints are exact, including differences too small to round normally.
inline bool excludes_zero(const Tensor& t,bool angular_difference=false) {
    if(t.nr<0 || t.nt<(angular_difference?1:0))return false;
    const int end=t.nt-(angular_difference?1:0);
    bool positive=true,negative=true;
    for(int i=0;i<=t.nr;++i)for(int j=0;j<=end;++j) {
        const auto a=t.b[i][j];
        if(!(a.lo<=a.hi))return false;
        if(angular_difference) {
            const auto b=t.b[i][j+1];
            if(!(b.lo<=b.hi))return false;
            positive=positive && b.lo>a.hi;
            negative=negative && b.hi<a.lo;
        } else {
            positive=positive && a.lo>0;
            negative=negative && a.hi<0;
        }
        if(!positive && !negative)return false;
    }
    return positive || negative;
}
// Search-order heuristic only: every split retains both children. Tensor
// differences already include each box's coordinate widths. Balance the
// directional variation of F and F_t separately, so a large row scale does
// not hide the other equation. Midpoints choose work; interval coefficients
// and the existing inclusion/exclusion proofs still certify every decision.
inline bool prefer_radial_split(const Tensor& t,double rw,double tw) {
    if(rw<tw/16)return false;
    if(rw>tw)return true;
    // Every subsequently accessed entry is filled below.
    std::array<std::array<double,7>,9> c,dt;
    for(int i=0;i<=t.nr;++i)for(int j=0;j<=t.nt;++j)
        c[i][j]=t.b[i][j].lo+(t.b[i][j].hi-t.b[i][j].lo)*.5;
    double ar=0,at=0,br=0,bt=0;
    for(int i=0;i<=t.nr;++i)for(int j=0;j<t.nt;++j) {
        dt[i][j]=c[i][j+1]-c[i][j];at=std::max(at,std::abs(dt[i][j]));
    }
    for(int i=0;i<t.nr;++i)for(int j=0;j<=t.nt;++j)
        ar=std::max(ar,std::abs(c[i+1][j]-c[i][j]));
    for(int i=0;i<t.nr;++i)for(int j=0;j<t.nt;++j)
        br=std::max(br,std::abs(dt[i+1][j]-dt[i][j]));
    for(int i=0;i<=t.nr;++i)for(int j=0;j<t.nt-1;++j)
        bt=std::max(bt,std::abs(dt[i][j+1]-dt[i][j]));
    ar*=t.nr;at*=t.nt;br*=t.nr;bt*=t.nt-1;
    const double a=ar+at,b=br+bt;
    if(!(a>0 && b>0 && std::isfinite(a) && std::isfinite(b)))return 4*rw>tw;
    return ar/a+br/b>1;
}
inline I hull(const Tensor& t){
    I out=t.b[0][0];
    for(int i=0;i<=t.nr;++i)for(int j=0;j<=t.nt;++j){
        if(t.b[i][j].lo!=t.b[i][j].lo || t.b[i][j].hi!=t.b[i][j].hi)return {-INFINITY,INFINITY};
        out.lo=std::min(out.lo,t.b[i][j].lo);out.hi=std::max(out.hi,t.b[i][j].hi);
    }
    return out;
}
// Hull of first differences needs only the extreme rounded differences.
// Round the final extrema outward once; monotonicity gives the same enclosure
// without materializing and clearing an entire derivative tensor.
inline I difference_hull(const Tensor& t,int axis) {
    double lo=INFINITY,hi=-INFINITY;
    for(int i=0;i<=t.nr-(axis==0);++i)for(int j=0;j<=t.nt-(axis==1);++j) {
        const I a=axis?t.b[i][j+1]:t.b[i+1][j],b=t.b[i][j];
        const double l=a.lo-b.hi,h=a.hi-b.lo;
        if(l!=l || h!=h)return {-INFINITY,INFINITY};
        lo=std::min(lo,l);hi=std::max(hi,h);
    }
    return {interval_detail::down(lo),interval_detail::up(hi)};
}
inline Tensor differences(const Tensor& t,int axis){
    Tensor out;out.nr=t.nr-(axis==0);out.nt=t.nt-(axis==1);
    for(int i=0;i<=out.nr;++i)for(int j=0;j<=out.nt;++j)
        out.b[i][j]=(axis?t.b[i][j+1]:t.b[i+1][j])-t.b[i][j];
    return out;
}
struct DerivativeHulls {I r,rt,tt;};
inline DerivativeHulls derivative_hulls(const Tensor& t) {
#if defined(__SSE2__)
    // A difference a-b in signed endpoints is a plus the swapped lanes
    // of b. Form each first difference once, retaining only adjacent
    // entries for mixed and second differences. Round their final hulls
    // once by monotonicity, as in difference_hull.
    if(t.nr==8 && t.nt==6) {
        constexpr double limit=std::numeric_limits<double>::max()/8;
        const auto valid_limit=_mm_set1_pd(limit);
        const auto sign=_mm_set1_pd(-0.);
        auto bounded=_mm_cmpeq_pd(sign,sign);
        auto hr=_mm_set1_pd(-INFINITY),hrt=hr,htt=hr;
        for(int i=0;i<=8;++i) {
            __m128d last_r{},last_t{};
            for(int j=0;j<=6;++j) {
                const auto b=signed_endpoints(t.b[i][j]);
                const auto abs_b=_mm_andnot_pd(sign,b);
                // The scalar interval operators preserve exact zero inputs.
                // Use their path for zero intervals and exceptional entries.
                const auto magnitude=_mm_max_pd(abs_b,_mm_shuffle_pd(abs_b,abs_b,1));
                bounded=_mm_and_pd(bounded,_mm_and_pd(_mm_cmpgt_pd(magnitude,_mm_setzero_pd()),
                    _mm_cmplt_pd(abs_b,valid_limit)));
                const auto neg=_mm_shuffle_pd(b,b,1);
                if(i<8) {
                    const auto dr=packed_up(_mm_add_pd(signed_endpoints(t.b[i+1][j]),neg));
                    hr=_mm_max_pd(hr,dr);
                    if(j)hrt=_mm_max_pd(hrt,_mm_add_pd(dr,_mm_shuffle_pd(last_r,last_r,1)));
                    last_r=dr;
                }
                if(j<6) {
                    const auto dt=packed_up(_mm_add_pd(signed_endpoints(t.b[i][j+1]),neg));
                    if(j)htt=_mm_max_pd(htt,_mm_add_pd(dt,_mm_shuffle_pd(last_t,last_t,1)));
                    last_t=dt;
                }
            }
        }
        if(_mm_movemask_pd(bounded)==3)
            return {endpoints(hr),endpoints(packed_up(hrt)),endpoints(packed_up(htt))};
    }
#endif
    const auto pr=differences(t,0),pt=differences(t,1);
    return {hull(pr),difference_hull(pr,1),difference_hull(pt,1)};
}
inline std::array<I,9> point_weights(int degree,I x) {
    std::array<I,9> xp{},yp{},out{};
    if(x.lo==.5 && x.hi==.5) {
        for(int j=0;j<=degree;++j)out[j]=I(double(angular_detail::choose(degree,j))/double(1u<<degree));
        return out;
    }
    xp[0]=yp[0]=I(1);const I y=I(1)-x;
    for(int j=1;j<=degree;++j){xp[j]=xp[j-1]*x;yp[j]=yp[j-1]*y;}
    for(int j=0;j<=degree;++j)out[j]=I(angular_detail::choose(degree,j))*xp[j]*yp[degree-j];
    return out;
}
inline std::array<I,7> radial_columns(const Tensor& t,I x) {
    // A polar midpoint in binary64 may be displaced slightly from the
    // exact Bernstein midpoint. Use exact dyadic weights for the radial
    // reduction and bound the displacement with |dP/dx| <= 2*n*max|b|.
    // The remaining angular reduction uses the original interval weights.
    if(t.nr==8 && t.nt==6 && x.lo>=.499999 && x.hi<=.500001 && x.lo<=x.hi) {
        constexpr double weights[]={1./256,8./256,28./256,56./256,70./256,
                                    56./256,28./256,8./256,1./256};
        constexpr double eps=std::numeric_limits<double>::epsilon();
        const double error_factor=32*(eps+std::max(std::abs(x.lo-.5),std::abs(x.hi-.5)));
        std::array<double,7> lo{},hi{},magnitude{};
        for(int i=0;i<9;++i)for(int j=0;j<7;++j) {
            const auto b=t.b[i][j];
            lo[j]+=weights[i]*b.lo;
            hi[j]+=weights[i]*b.hi;
            magnitude[j]=std::max(magnitude[j],std::max(std::abs(b.lo),std::abs(b.hi)));
        }
        std::array<I,7> columns;
        bool bounded=true;
        for(int j=0;j<7;++j) {
            // Nine products and eight additions of a convex combination.
            const double pad=magnitude[j]*error_factor+32*std::numeric_limits<double>::denorm_min();
            bounded=bounded && std::isfinite(lo[j]) && std::isfinite(hi[j]) &&
                magnitude[j]<std::numeric_limits<double>::max()/4;
            columns[j]={interval_detail::down(lo[j]-pad),interval_detail::up(hi[j]+pad)};
        }
        if(bounded)return columns;
    }
    const auto rx=point_weights(t.nr,x);
    std::array<I,7> columns{};
    for(int j=0;j<=t.nt;++j)for(int i=0;i<=t.nr;++i)columns[j]=columns[j]+rx[i]*t.b[i][j];
    return columns;
}
inline std::pair<I,I> centre_value_theta(const Tensor& t,I x,I y,I theta_width) {
    const auto columns=radial_columns(t,x);
    const auto ty=point_weights(t.nt,y),dy=point_weights(t.nt-1,y);
    I value,theta;
    for(int j=0;j<=t.nt;++j)value=value+ty[j]*columns[j];
    for(int j=0;j<t.nt;++j)theta=theta+dy[j]*(columns[j+1]-columns[j]);
    return {value,theta*I(t.nt)*inverse(theta_width)};
}
inline Tensor derivative(const Tensor& t,int axis,I width){
    Tensor out;out.nr=t.nr-(axis==0);out.nt=t.nt-(axis==1);
    const I scale=I(axis?t.nt:t.nr)*inverse(width);
    for(int i=0;i<=out.nr;++i)for(int j=0;j<=out.nt;++j)
        out.b[i][j]=scale*((axis?t.b[i][j+1]:t.b[i+1][j])-t.b[i][j]);
    return out;
}
inline I evaluate(const Tensor& t,I x,I y){
    std::array<I,9> rows{};
    for(int i=0;i<=t.nr;++i){auto row=t.b[i];
        for(int n=t.nt;n>0;--n)for(int j=0;j<n;++j)row[j]=(I(1)-y)*row[j]+y*row[j+1];
        rows[i]=row[0];}
    for(int n=t.nr;n>0;--n)for(int j=0;j<n;++j)rows[j]=(I(1)-x)*rows[j]+x*rows[j+1];
    return rows[0];
}
inline Tensor restrict(Tensor tensor,double rlo,double rhi,double tlo,double thi,
                       double nrlo,double nrhi,double ntlo,double nthi){
    if(nrhi<rhi)tensor=split(tensor,0,fraction(nrhi,rlo,rhi)).first;
    if(nrlo>rlo)tensor=split(tensor,0,fraction(nrlo,rlo,nrhi)).second;
    if(nthi<thi)tensor=split(tensor,1,fraction(nthi,tlo,thi)).first;
    if(ntlo>tlo)tensor=split(tensor,1,fraction(ntlo,tlo,nthi)).second;
    return tensor;
}

}
