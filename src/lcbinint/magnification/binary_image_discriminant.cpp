#include "lcbinint/magnification/binary_image_discriminant.hpp"
#include "lcbinint/math/outward_interval.hpp"
#include "lcbinint/math/filtered_ball.hpp"

#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <unordered_map>
#include <cstdint>
#include "lcbinint/magnification/caustic_factor_generated.hpp"

namespace lcbinint::magnification {
namespace {
using I = math::OutwardInterval;

bool arithmetic_supported()
{
#ifdef __FAST_MATH__
    return false;
#endif
    if (!std::numeric_limits<double>::is_iec559
        || std::numeric_limits<double>::digits!=53
        || std::numeric_limits<double>::radix!=2)
        return false;
    // Include a gradual-underflow check; changing the hardware FP environment
    // must not silently turn an interval sign into a certificate.
    volatile double smallest_normal=std::numeric_limits<double>::min();
    volatile double smallest_subnormal=std::numeric_limits<double>::denorm_min();
    if (std::fegetround()!=FE_TONEAREST || smallest_normal*0.5==0.0
        || smallest_subnormal+smallest_subnormal==0.0)
        return false;
    return true;
}


bool negative(I a) { return a.hi<0.0; }
bool negative(math::FilteredBall a) { return a.negative(); }

// Geometry coefficients retain their rounding enclosure. Cache keys are exact
// native geometry values; orbital motion therefore invalidates the cache.
template<class T> struct CausticGeometry {
    double separation=0, mass_ratio=0;
    T l{}, nu{};
    std::array<std::array<T,9>,17> coefficients{};
    void prepare(double s,double q_input) {
        if (separation==s && mass_ratio==q_input) return;
        separation=s; mass_ratio=q_input;
        l=T(std::abs(s));
        const T q=q_input<1 ? T(q_input) : T(1)/T(q_input);
        nu=q/(T(1)+q);
        prepare_coefficients(l,nu);
    }
    void prepare_coefficients(T length,T fraction) {
        l=length;nu=fraction;
        std::array<T,25> lp{}; std::array<T,9> np{};
        lp[8]=np[0]=T(1);
        for (int i=9;i<25;++i) lp[i]=lp[i-1]*l;
        for (int i=7;i>=0;--i) lp[i]=lp[i+1]/l;
        for (int i=1;i<9;++i) np[i]=np[i-1]*nu;
        coefficients={};
        for (const auto& t:detail::caustic_terms)
            coefficients[t.x][t.y2]=coefficients[t.x][t.y2]+T(t.coefficient)*lp[t.l+8]*np[t.nu];
    }
    std::array<T,2> gradient(T x,T y) const {
        const T y2=math::square(y),dy2=T(2)*y;
        T value{},gx{},gy{};
        for (int i=16;i>=0;--i) {
            T row=coefficients[i][(16-i)/2],row_y{};
            for (int j=(16-i)/2-1;j>=0;--j) {
                row_y=row_y*y2+row*dy2;
                row=row*y2+coefficients[i][j];
            }
            gx=gx*x+value; gy=gy*x+row_y;
            value=value*x+row;
        }
        return {gx,gy};
    }
    T directional_derivative(T x,T y,T dx,T dy) const {
        const T y2=math::square(y), dy2=T(2)*y*dy;
        T value{}, derivative{};
        for (int i=16;i>=0;--i) {
            T row=coefficients[i][(16-i)/2], row_derivative{};
            for (int j=(16-i)/2-1;j>=0;--j) {
                row_derivative=row_derivative*y2+row*dy2;
                row=row*y2+coefficients[i][j];
            }
            derivative=derivative*x+value*dx+row_derivative;
            value=value*x+row;
        }
        return derivative;
    }
    T evaluate(T x,T y) const {
        const T y2=math::square(y);
        T result{};
        for (int i=16;i>=0;--i) {
            T row=coefficients[i][(16-i)/2];
            for (int j=(16-i)/2-1;j>=0;--j) row=row*y2+coefficients[i][j];
            result=result*x+row;
        }
        return result;
    }
};
template<class T> CausticGeometry<T>& caustic_geometry(double s,double q) {
    thread_local CausticGeometry<T> geometry;
    geometry.prepare(s,q);
    return geometry;
}
template<class T> int caustic_sign(double s,double q,T source_x,T source_y) {
    auto& geometry=caustic_geometry<T>(s,q);
    const T x=(q<1 ? source_x : -source_x)+geometry.l*geometry.nu
        -(geometry.l-T(1)/geometry.l);
    const T value=geometry.evaluate(x,source_y);
    if (!value.finite() || value.contains_zero()) return 0;
    return value.minimum_magnitude()>0 ? (negative(value) ? -1 : 1) : 0;
}
bool caustic_factor_enabled() {
    const char* value=std::getenv("LCBININT_CAUSTIC_FACTOR");
    return value && value[0]=='1';
}

// Asada, Kasai & Kasai (2002), Eqs. (2.8)--(2.13), expressed in Cartesian
// source coordinates x=rho*C, y=rho*S relative to the primary at (0,0), with
// secondary mass fraction nu at (l,0). Descending powers of cot(image angle).
template<class I>
std::array<I, 6> real_quintic(I l, I nu, I x, I y)
{
    const I x2 = math::square(x), y2 = math::square(y);
    const I r2 = x2+y2, l2 = math::square(l), l3 = l2*l, l4 = math::square(l2);
    return {{
        nu*l*y2*y,
        y2*(I(1)+I(2)*l*x-l2*(I(2)-r2)-I(2)*l3*x+l4
            -nu*(I(5)*l*x-I(4)*l2)),
        y*(-I(2)*x-I(4)*l*(x2-y2)+l2*(I(4)*x-I(2)*r2*x)
           +I(4)*l3*x2-I(2)*l4*x
           +nu*(l*(I(2)+I(8)*x2-I(2)*y2)-I(10)*l2*x+I(2)*l3)),
        r2+l*(I(2)*x2*x-I(10)*x*y2)+l2*(-I(2)*x2+I(2)*y2+math::square(r2))
           -I(2)*l3*x*r2+l4*r2
           +nu*(l*(-I(2)*x-I(4)*x2*x+I(6)*x*y2)+I(6)*l2*x2-I(2)*l3*x)
           +math::square(nu)*l2,
        y*(-I(2)*x+I(8)*l*x2-l2*(I(4)*x+I(2)*r2*x)
           +I(4)*l3*x2-I(2)*l4*x
           +nu*(l*(I(2)-I(4)*x2+y2)-I(2)*l2*x+I(2)*l3)),
        x2-I(2)*l*x2*x+l2*(I(2)*x2+r2*x2)-I(2)*l3*x2*x+l4*x2
           +nu*(-l*(I(2)*x+x*y2)+I(2)*l2*x2-I(2)*l3*x)+math::square(nu)*l2,
    }};
}

// Eq. (2.22), W/y^2. W=0 is a coordinate reduction artifact, not a caustic.
template<class I>
I reduction_factor(I l, I nu, I x, I y)
{
    const I l2=math::square(l), l4=math::square(l2), r2=math::square(x)+math::square(y);
    const I c=I(1)-l4-I(2)*l2*nu;
    return l2*math::square(r2)-I(4)*x*l2*l*r2-r2*c
        +I(4)*math::square(x)*l4+I(2)*x*l*c-l2*nu*(I(2)-I(2)*l2-nu);
}

// Bezout(p,p') has determinant leading_coefficient^2 * discriminant(p).
// Construct it directly from polynomial coefficients, without solving roots.
// Interval Gaussian elimination only reports a sign if every pivot is separated
// from zero. Dependency widening may refuse perfectly regular cases.
template<class I>
int discriminant_sign(std::array<I,6> descending)
{
    double scale=0.0;
    for (const auto& a: descending) {
        if (!a.finite()) return 0;
        scale=std::max(scale,a.magnitude());
    }
    if (!(scale>0.0)) return 0;
    std::array<I,6> p{}, derivative{};
    int exponent=0;
    std::frexp(scale,&exponent);
    const double reciprocal=std::scalbn(1.0,-exponent);
    for (int i=0;i<6;++i) {
        p[i]=std::isfinite(reciprocal) && reciprocal>0.0
            ? descending[5-i]*I(reciprocal) : descending[5-i]/I(scale);
    }
    for (int i=0;i<5;++i) derivative[i]=I(i+1)*p[i+1];
    std::array<std::array<I,5>,5> b{};
    for (int i=1;i<=5;++i) {
        for (int j=0;j<i;++j) {
            const I term=p[i]*derivative[j]-p[j]*derivative[i];
            for (int k=0;k<i-j;++k) b[i-1-k][j+k]=b[i-1-k][j+k]+term;
        }
    }
    int sign=1;
    for (int k=0;k<5;++k) {
        int pivot=k;
        for (int i=k;i<5;++i) {
            if (!b[i][k].finite()) return 0;
            if (b[i][k].minimum_magnitude()>b[pivot][k].minimum_magnitude()) pivot=i;
        }
        if (b[pivot][k].contains_zero()) return 0;
        if (pivot!=k) { std::swap(b[pivot],b[k]); sign=-sign; }
        sign *= negative(b[k][k]) ? -1 : 1;
        for (int i=k+1;i<5;++i) {
            const I ratio=b[i][k]/b[k][k];
            for (int j=k+1;j<5;++j) b[i][j]=b[i][j]-ratio*b[k][j];
        }
    }
    return sign;
}
} // namespace

BinaryImageCount binary_image_count_caustic_filter(
    double separation,double mass_ratio,SourcePosition source)
{
    if (!arithmetic_supported()) return {-1,DiscriminantStatus::unsupported_arithmetic};
    if (!std::isfinite(separation) || !std::isfinite(mass_ratio)
        || separation==0 || mass_ratio==0 || !std::isfinite(source.x) || !std::isfinite(source.y))
        return {-1,DiscriminantStatus::invalid_geometry};
    using B=math::FilteredBall;
    const int sign=caustic_sign(separation,std::abs(mass_ratio),B(source.x),B(source.y));
    if (!sign) return {-1,DiscriminantStatus::uncertain_determinant};
    return {sign<0 ? 5 : 3,DiscriminantStatus::resolved,true};
}

BinaryImageCount binary_image_count_rounded_geometry(double separation,double mass_ratio,SourcePosition source)
{
    if (!arithmetic_supported()) return {-1,DiscriminantStatus::unsupported_arithmetic};
    if (!std::isfinite(separation) || !std::isfinite(mass_ratio) || separation==0 || mass_ratio==0 ||
        !std::isfinite(source.x) || !std::isfinite(source.y)) return {-1,DiscriminantStatus::invalid_geometry};
    using B=math::FilteredBall;
    struct Geometry {
        double s=0,q=0;
        I scale,length,fraction,shift,a;
        CausticGeometry<B> factor;
    };
    static thread_local Geometry geometry;
    const double qi=std::abs(mass_ratio),s=std::abs(separation);
    const auto ball=[](I x) {
        const double centre=x.lo*.5+x.hi*.5;
        return B(centre,std::max((I(centre)-I(x.lo)).hi,(I(x.hi)-I(centre)).hi));
    };
    if (geometry.s!=s || geometry.q!=qi) {
        geometry.s=s;geometry.q=qi;geometry.a=I(qi<1 ? -s : s);
        // This parameter box contains both the exact-input masses and the
        // existing native rounded mapper, including its rounded origin.
        const I q=qi<1 ? I(qi) : I(1)/I(qi),m1=I(1)/(I(1)+q),m2=q*m1;
        const I mass=m1+m2;
        geometry.scale=I(std::max(0.0,math::down(std::sqrt(mass.lo))),math::up(std::sqrt(mass.hi)));
        geometry.length=I(s)/geometry.scale;geometry.fraction=m2/mass;
        geometry.shift=geometry.a*m1;
        geometry.factor.prepare_coefficients(ball(geometry.length),ball(geometry.fraction));
    }
    const auto& g=geometry;
    const I x=I(qi<1 ? 1 : -1)*(I(source.x)+g.shift-g.a)/g.scale,y=I(source.y)/g.scale;
    const B chart=ball(x)-(ball(g.length)-B(1)/ball(g.length));
    const B value=g.factor.evaluate(chart,ball(y));
    if (value.finite() && !value.contains_zero())
        return {value.negative() ? 5 : 3,DiscriminantStatus::resolved,true};
    if (y.contains_zero()) return {-1,DiscriminantStatus::symmetry_axis};
    const I reduction=reduction_factor(g.length,g.fraction,x,y);
    if (!reduction.finite() || reduction.contains_zero()) return {-1,DiscriminantStatus::coordinate_degeneracy};
    const int sign=discriminant_sign(real_quintic(g.length,g.fraction,x,y));
    return sign ? BinaryImageCount{sign>0 ? 5 : 3,DiscriminantStatus::resolved} : BinaryImageCount{-1,DiscriminantStatus::uncertain_determinant};
}

BinaryImageCount BinaryCausticRay::count(math::FilteredBall parameter) const
{
    if (!valid || !arithmetic_supported() || !parameter.finite())
        return {-1,DiscriminantStatus::uncertain_determinant};
    auto value=coefficients[16];
    for (int i=15;i>=0;--i) value=value*parameter+coefficients[i];
    if (!value.finite() || value.contains_zero())
        return {-1,DiscriminantStatus::uncertain_determinant};
    return {value.negative() ? 5 : 3,DiscriminantStatus::resolved,true};
}

BinaryCausticRay binary_caustic_ray(double separation,double mass_ratio,
    SourcePosition first,SourcePosition last,SourcePosition residual_radius)
{
    using B=math::FilteredBall;
    BinaryCausticRay result;
    if (!arithmetic_supported() || !std::isfinite(separation) || !std::isfinite(mass_ratio)
        || separation==0 || mass_ratio==0
        || !std::isfinite(first.x) || !std::isfinite(first.y)
        || !std::isfinite(last.x) || !std::isfinite(last.y)
        || !std::isfinite(residual_radius.x) || !std::isfinite(residual_radius.y)
        || residual_radius.x<0 || residual_radius.y<0) return result;
    const double q=std::abs(mass_ratio),reflection=q<1 ? 1 : -1;
    auto& g=caustic_geometry<B>(separation,q);
    const B x0=B(reflection)*B(first.x,residual_radius.x)+g.l*g.nu-(g.l-B(1)/g.l);
    const B x1=B(reflection)*(B(last.x)-B(first.x));
    const B y0(first.y,residual_radius.y),y1=B(last.y)-B(first.y);
    const std::array<B,3> y2{y0*y0,B(2)*y0*y1,y1*y1};
    // Nested polynomial Horner substitution. Multiplication is only by an
    // affine x or a quadratic y^2, so no general polynomial product is needed.
    std::array<B,17> value{};
    int value_degree=0;
    for (int i=16;i>=0;--i) {
        std::array<B,17> row{};
        const int max_y=(16-i)/2;
        row[0]=g.coefficients[i][max_y];
        int row_degree=0;
        for (int j=max_y-1;j>=0;--j) {
            std::array<B,17> next{};
            for (int k=0;k<=row_degree;++k)
                for (int d=0;d<=2;++d) next[k+d]=next[k+d]+row[k]*y2[d];
            next[0]=next[0]+g.coefficients[i][j];
            row=next; row_degree+=2;
        }
        std::array<B,17> next{};
        for (int k=0;k<=value_degree;++k) {
            next[k]=next[k]+value[k]*x0;
            if (k<16) next[k+1]=next[k+1]+value[k]*x1;
        }
        value_degree=std::min(16,value_degree+1);
        for (int k=0;k<=row_degree;++k) next[k]=next[k]+row[k];
        value=next;
    }
    result.coefficients=value;
    result.valid=true;
    for (const auto& c:value) result.valid=result.valid && c.finite();
    return result;
}

BinaryCausticGradient binary_caustic_gradient_box(
    double separation,double mass_ratio,SourcePosition lower,SourcePosition upper)
{
    if (!arithmetic_supported() || !std::isfinite(separation) || !std::isfinite(mass_ratio)
        || separation==0 || mass_ratio==0 || !std::isfinite(lower.x) || !std::isfinite(lower.y)
        || !std::isfinite(upper.x) || !std::isfinite(upper.y) || lower.x>upper.x || lower.y>upper.y)
        return {};
    using B=math::FilteredBall;
    const auto enclose=[](double lo,double hi) {
        const double mid=lo*0.5+hi*0.5;
        return B(mid,std::max((I(hi)-I(mid)).hi,(I(mid)-I(lo)).hi));
    };
    const double q=std::abs(mass_ratio), reflection=q<1 ? 1 : -1;
    auto& g=caustic_geometry<B>(separation,q);
    const B x=B(reflection)*enclose(lower.x,upper.x)+g.l*g.nu-(g.l-B(1)/g.l);
    const auto gradient=g.gradient(x,enclose(lower.y,upper.y));
    return {B(reflection)*gradient[0],gradient[1],gradient[0].finite() && gradient[1].finite()};
}

BinarySourceBox binary_source_box(double separation,double mass_ratio,
    SourcePosition source,double source_radius)
{
    BinarySourceBox result;
    if (!arithmetic_supported() || !std::isfinite(separation) || !std::isfinite(mass_ratio)
        || separation==0 || mass_ratio==0 || !std::isfinite(source_radius) || source_radius<=0
        || !std::isfinite(source.x) || !std::isfinite(source.y)) return result;
    struct Key {
        std::int64_t x,y; int exponent;
        bool operator==(const Key& b) const {return x==b.x && y==b.y && exponent==b.exponent;}
    };
    struct Hash {
        std::size_t operator()(const Key& k) const {
            auto h=std::hash<std::int64_t>{}(k.x);
            h^=std::hash<std::int64_t>{}(k.y)+0x9e3779b9+(h<<6)+(h>>2);
            return h^(std::hash<int>{}(k.exponent)+0x9e3779b9+(h<<6)+(h>>2));
        }
    };
    struct Cache {
        double separation=0,q=0;
        std::unordered_map<Key,int,Hash> cells;
    };
    thread_local Cache cache;
    const double q=std::abs(mass_ratio);
    if (cache.separation!=separation || cache.q!=q) {
        cache.cells.clear();cache.separation=separation;cache.q=q;
    }
    using B=math::FilteredBall;
    const double reflection=q<1 ? 1 : -1;
    auto& g=caustic_geometry<B>(separation,q);
    const int base=std::ilogb(source_radius)-1;
    for (int level=0;level<12;++level) {
        const int exponent=base-level;
        const double side=std::ldexp(1.0,exponent);
        if (!std::isfinite(side) || !(side*0.5>0)) break;
        const double ix=std::floor(source.x/side),iy=std::floor(source.y/side);
        if (!std::isfinite(ix) || !std::isfinite(iy) || std::abs(ix)>=0x1p49 || std::abs(iy)>=0x1p49) break;
        const Key key{static_cast<std::int64_t>(ix),static_cast<std::int64_t>(iy),exponent};
        const SourcePosition lower{ix*side,iy*side},upper{(ix+1)*side,(iy+1)*side};
        // Dyadic endpoints are exact here. Explicit containment also rejects
        // an underflowed coordinate quotient or an unsupported edge case.
        if (!std::isfinite(lower.x) || !std::isfinite(lower.y) || !std::isfinite(upper.x) || !std::isfinite(upper.y)
            || !(lower.x<=source.x && source.x<=upper.x && lower.y<=source.y && source.y<=upper.y)) break;
        auto found=cache.cells.find(key);
        int count=-1;
        if (found!=cache.cells.end()) {
            ++result.cache_hits;count=found->second;
        } else {
            if (result.enclosure_evaluations!=0) continue;
            const double mx=lower.x+side*0.5,my=lower.y+side*0.5;
            const B half(0,side*0.5);
            const B xc=B(reflection)*B(mx)+g.l*g.nu-(g.l-B(1)/g.l);
            const auto gradient=g.gradient(xc+B(reflection)*half,B(my)+half);
            const B value=g.evaluate(xc,B(my))+gradient[0]*B(reflection)*half+gradient[1]*half;
            if (value.finite() && !value.contains_zero()) count=value.negative() ? 5 : 3;
            // Bound retained memory. Eviction only loses acceleration; it
            // cannot promote an unresolved cell to a certificate.
            if (cache.cells.size()>=65536) cache.cells.erase(cache.cells.begin());
            cache.cells.emplace(key,count);
            ++result.enclosure_evaluations;
        }
        if (count>=0) {
            result.lower=lower;result.upper=upper;result.image_count=count;
            return result;
        }
    }
    return result;
}

BinaryImageCount binary_image_count_discriminant(
    double separation, double mass_ratio, SourcePosition source, bool allow_filter)
{
    if (!arithmetic_supported()) return {-1,DiscriminantStatus::unsupported_arithmetic};
    if (!std::isfinite(separation) || !std::isfinite(mass_ratio)
        || !std::isfinite(source.x) || !std::isfinite(source.y)
        || separation==0.0 || mass_ratio==0.0)
        return {-1,DiscriminantStatus::invalid_geometry};
    const double q_input=std::abs(mass_ratio);
    if (caustic_factor_enabled()) {
        if (allow_filter) {
            const int sign=caustic_sign(separation,q_input,math::FilteredBall(source.x),math::FilteredBall(source.y));
            if (sign) return {sign<0 ? 5 : 3,DiscriminantStatus::resolved,true};
        }
        const int sign=caustic_sign(separation,q_input,I(source.x),I(source.y));
        if (sign) return {sign<0 ? 5 : 3,DiscriminantStatus::resolved};
        // Cancellation in the expanded factor can be severe near planetary
        // caustics. Retain the independent quintic enclosure as fallback.
    }
    if (source.y==0.0) return {-1,DiscriminantStatus::symmetry_axis};
    // Cheap forward-error enclosure first. All transforms and coefficients
    // are evaluated in the same enclosure; a bare double determinant is never
    // accepted using a heuristic tolerance.
    if (allow_filter) {
        using B = math::FilteredBall;
        const B qb=q_input<1.0 ? B(q_input) : B(1)/B(q_input);
        const B nub=qb/(B(1)+qb), lb(std::abs(separation));
        const B xb=B(q_input<1.0 ? source.x : -source.x)+lb*nub, yb(source.y);
        const B wb=reduction_factor(lb,nub,xb,yb);
        if (wb.finite() && !wb.contains_zero()) {
            const int filtered_sign=discriminant_sign(real_quintic(lb,nub,xb,yb));
            if (filtered_sign!=0)
                return {filtered_sign>0 ? 5 : 3,DiscriminantStatus::resolved,true};
        }
    }
    const I q=q_input<1.0 ? I(q_input) : I(1)/I(q_input);
    const I nu=q/(I(1)+q), l(std::abs(separation));
    // Native mapper: primary at a=+-s, secondary at zero, source shifted by
    // a*m1. Translate to the primary and reflect x when a>0. y reflection is
    // immaterial to image count because the lens is symmetric about the axis.
    const I x=I(q_input<1.0 ? source.x : -source.x)+l*nu, y(source.y);
    const I w=reduction_factor(l,nu,x,y);
    if (!w.finite() || w.contains_zero())
        return {-1,DiscriminantStatus::coordinate_degeneracy};
    const int sign=discriminant_sign(real_quintic(l,nu,x,y));
    if (sign==0) return {-1,DiscriminantStatus::uncertain_determinant};
    return {sign>0 ? 5 : 3,DiscriminantStatus::resolved};
}

BinaryImageCount binary_image_count_box(
    double separation, double mass_ratio, SourcePosition lower, SourcePosition upper)
{
    if (!arithmetic_supported()) return {-1,DiscriminantStatus::unsupported_arithmetic};
    if (!std::isfinite(separation) || !std::isfinite(mass_ratio)
        || !std::isfinite(lower.x) || !std::isfinite(lower.y)
        || separation==0.0 || mass_ratio==0.0)
        return {-1,DiscriminantStatus::invalid_geometry};
    if (!std::isfinite(upper.x) || !std::isfinite(upper.y)
        || lower.x>upper.x || lower.y>upper.y)
        return {-1,DiscriminantStatus::invalid_geometry};
    const double q_input=std::abs(mass_ratio);
    const I q=q_input<1.0 ? I(q_input) : I(1)/I(q_input);
    const I nu=q/(I(1)+q), l(std::abs(separation));
    const I x=(q_input<1.0 ? I(lower.x,upper.x) : I(-upper.x,-lower.x))+l*nu;
    if (caustic_factor_enabled()) {
        using B=math::FilteredBall;
        const auto enclose=[](double lo,double hi) {
            const double mid=lo*0.5+hi*0.5;
            const I radius=I(hi)-I(mid), other=I(mid)-I(lo);
            return B(mid,std::max(radius.hi,other.hi));
        };
        const int sign=caustic_sign(separation,q_input,enclose(lower.x,upper.x),enclose(lower.y,upper.y));
        if (sign) return {sign<0 ? 5 : 3,DiscriminantStatus::resolved,true};
        // The old rectangle determinant resolved no links in the pilot;
        // declining a box here avoids seven expensive futile eliminations.
        return {-1,DiscriminantStatus::uncertain_determinant};
    }
    if (lower.y<=0.0 && upper.y>=0.0) return {-1,DiscriminantStatus::symmetry_axis};
    const I y(lower.y,upper.y), w=reduction_factor(l,nu,x,y);
    if (!w.finite() || w.contains_zero()) return {-1,DiscriminantStatus::coordinate_degeneracy};
    const int sign=discriminant_sign(real_quintic(l,nu,x,y));
    if (!sign) return {-1,DiscriminantStatus::uncertain_determinant};
    return {sign>0 ? 5 : 3,DiscriminantStatus::resolved};
}

bool binary_image_link_verified(double separation, double mass_ratio,
    SourcePosition first, SourcePosition last, int expected_count, int* boxes_used)
{
    if (boxes_used) *boxes_used=0;
    if (expected_count!=3 && expected_count!=5) return false;
    const I dx=I(last.x)-I(first.x),dy=I(last.y)-I(first.y);
    const auto check=[&](auto&& self,double left,double right,int depth)->bool {
        const I t(left,right),x=I(first.x)+t*dx,y=I(first.y)+t*dy;
        if (boxes_used) ++*boxes_used;
        const auto result=binary_image_count_box(separation,mass_ratio,{x.lo,y.lo},{x.hi,y.hi});
        if (result.image_count<0 && caustic_factor_enabled() && arithmetic_supported()) {
            // Mean value enclosure along the actual segment, retaining the
            // direction in dK/dt instead of enlarging to an unrelated rectangle.
            using B=math::FilteredBall;
            const auto ball=[](I v) {
                const double mid=v.lo*0.5+v.hi*0.5;
                return B(mid,std::max((I(v.hi)-I(mid)).hi,(I(mid)-I(v.lo)).hi));
            };
            const double q=std::abs(mass_ratio),mid=0.5*(left+right);
            auto& g=caustic_geometry<B>(separation,q);
            const B shift=g.l*g.nu-(g.l-B(1)/g.l);
            const B bx=(q<1 ? ball(x) : -ball(x))+shift, by=ball(y);
            const B bdx=q<1 ? ball(dx) : -ball(dx),bdy=ball(dy);
            const B mx=(q<1 ? ball(I(first.x)+I(mid)*dx) : -ball(I(first.x)+I(mid)*dx))+shift;
            const B my=ball(I(first.y)+I(mid)*dy);
            const B center=g.evaluate(mx,my);
            const B derivative=g.directional_derivative(bx,by,bdx,bdy);
            const B enclosure=center+B(0,0.5*(right-left))*derivative;
            if (enclosure.finite() && !enclosure.contains_zero()) {
                const int count=enclosure.negative() ? 5 : 3;
                return count==expected_count;
            }
        }
        if (result.image_count==expected_count) return true;
        if (result.image_count>=0 || depth==2) return false;
        const double mid=0.5*(left+right); // dyadic endpoints, exact at this depth
        return self(self,left,mid,depth+1) && self(self,mid,right,depth+1);
    };
    return check(check,0.0,1.0,0);
}

const char* discriminant_status_name(DiscriminantStatus status)
{
    switch(status) {
    case DiscriminantStatus::resolved: return "resolved";
    case DiscriminantStatus::invalid_geometry: return "invalid_geometry";
    case DiscriminantStatus::symmetry_axis: return "symmetry_axis";
    case DiscriminantStatus::coordinate_degeneracy: return "coordinate_degeneracy";
    case DiscriminantStatus::uncertain_determinant: return "uncertain_determinant";
    case DiscriminantStatus::unsupported_arithmetic: return "unsupported_arithmetic";
    }
    return "invalid_geometry";
}
} // namespace lcbinint::magnification
