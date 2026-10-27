#pragma once
#include "geometry.hpp"
// Reuse the incumbent outward arithmetic and floating-point environment gate.
// No D14 polynomial, root solver or support planner is executed here.
#include "interval.hpp"
#include "radial_polynomial.hpp"
#include "physical_jet.hpp"
#include "point_polynomial.hpp"
#include <optional>
#include <memory>
#include <vector>

#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math","fp-contract=off")
#endif
namespace lcbinint::algebraic_boundary::triple {
namespace angular_detail {
namespace enclosure=interval_detail;
template<class T> using I=enclosure::Interval<T>;
template<class T> using Polynomial=std::array<I<T>,7>;
inline int choose(int n,int k) {
    int out=1;for(int j=1;j<=k;++j)out=out*(n-j+1)/j;return out;
}
template<class T> Polynomial<T> chart(Polynomial<T> p,bool reciprocal_chart) {
    if(!reciprocal_chart)return p;
    Polynomial<T> out;
    for(int j=0;j<7;++j)out[j]=j%2?-p[6-j]:p[6-j];
    return out;
}
// Power coefficients on [lo,hi] to degree-six Bernstein coefficients on [0,1].
// All chart endpoints are dyadic binary64 numbers, exact in both tiers.
template<class T> Polynomial<T> restrict(const Polynomial<T>& p,double lo,double hi) {
    Polynomial<T> power{},out{};
    const I<T> a(lo),width=I<T>(hi)-a;
    std::array<I<T>,7> ap{},wp{};ap[0]=wp[0]=I<T>(1);
    for(int j=1;j<7;++j){ap[j]=ap[j-1]*a;wp[j]=wp[j-1]*width;}
    for(int k=0;k<7;++k)for(int j=k;j<7;++j)
        power[k]=power[k]+p[j]*I<T>(choose(j,k))*ap[j-k]*wp[k];
    for(int i=0;i<7;++i)for(int k=0;k<=i;++k) {
        const T ratio=T(choose(i,k))/T(choose(6,k));
        const I<T> weight(enclosure::down(ratio),enclosure::up(ratio));
        out[i]=out[i]+power[k]*weight;
    }
    return out;
}
template<class T> std::pair<Polynomial<T>,Polynomial<T>> split(Polynomial<T> b) {
    Polynomial<T> left{},right{};left[0]=b[0];right[6]=b[6];
    for(int level=1;level<=6;++level) {
        for(int j=0;j<=6-level;++j)b[j]=(b[j]+b[j+1])*I<T>(.5);
        left[level]=b[0];right[6-level]=b[6-level];
    }
    return {left,right};
}
template<class T> int variations(const Polynomial<T>& b) {
    int previous=0,n=0;
    for(auto x:b) {
        const int sign=x.sign();if(sign==2)return -1;
        if(sign && previous && sign!=previous)++n;
        if(sign)previous=sign;
    }
    return n;
}
template<class T> I<T> power_value(const Polynomial<T>& p,int degree,I<T> x) {
    I<T> y=p[degree];for(int j=degree-1;j>=0;--j)y=y*x+p[j];return y;
}
// A complete real-root count by derivative monotonicity. Every critical
// interval is enclosed recursively and must exclude a zero of its parent.
// Ambiguous signs retain the general Bernstein fallback below.
template<class T> bool monotone_roots(const Polynomial<T>& p,int degree,double lo,double hi,
        double tolerance,std::vector<std::pair<double,double>>& roots,int& work,int budget) {
    if(++work>budget)return false;
    while(degree>0 && p[degree].zero())--degree;
    if(!p[degree].valid() || p[degree].sign()==2)return false;
    if(degree==0)return p[0].sign()==1 || p[0].sign()==-1;
    Polynomial<T> derivative{};
    for(int j=1;j<=degree;++j)derivative[j-1]=I<T>(j)*p[j];
    std::vector<std::pair<double,double>> critical;
    if(degree>1 && !monotone_roots(derivative,degree-1,lo,hi,
            std::max(tolerance,1e-9),critical,work,budget))return false;
    for(auto band:critical) {
        auto v=power_value(p,degree,I<T>(T(band.first),T(band.second)));
        if(v.sign()!=1 && v.sign()!=-1)return false;
    }
    double left=lo;
    for(std::size_t k=0;k<=critical.size();++k) {
        double right=k<critical.size()?critical[k].first:hi;
        auto fl=power_value(p,degree,I<T>(left)),fr=power_value(p,degree,I<T>(right));
        int sl=fl.sign(),sr=fr.sign();
        if((sl!=1 && sl!=-1) || (sr!=1 && sr!=-1))return false;
        if(sl!=sr) {
            double a=left,b=right;
            for(int iteration=0;iteration<64 && b-a>tolerance;++iteration) {
                if(++work>budget)return false;
                const double m=a+(b-a)*.5;
                const auto fm=power_value(p,degree,I<T>(m));
                const auto d=power_value(derivative,degree-1,I<T>(T(a),T(b)));
                bool contracted=false;
                if(d.valid() && (d.lo>0 || d.hi<0)) {
                    const I<T> inv(enclosure::down(T(1)/d.hi),enclosure::up(T(1)/d.lo));
                    const auto n=I<T>(m)-fm*inv;
                    double l=double(n.lo),h=double(n.hi);
                    if(T(l)>n.lo)l=std::nextafter(l,-INFINITY);
                    if(T(h)<n.hi)h=std::nextafter(h,INFINITY);
                    l=std::max(a,l);h=std::min(b,h);
                    if(h>l && h-l<.75*(b-a)){a=l;b=h;contracted=true;}
                }
                if(!contracted) {
                    const int sm=fm.sign();
                    if(sm!=1 && sm!=-1)return false;
                    // Recover a sign bracket after a Newton contraction.
                    const int sa=power_value(p,degree,I<T>(a)).sign();
                    if(sa!=1 && sa!=-1)return false;
                    if(sa==sm)a=m;else b=m;
                }
            }
            if(!(b>a) || b-a>tolerance)return false;
            roots.emplace_back(a,b);
        }
        if(k<critical.size())left=critical[k].second;
    }
    return true;
}
template<class T=double> inline bool contract_physical(double radius,const Geometry& p,bool chart,double& lo,double& hi,
                               double tolerance) {
    for(int iteration=0;iteration<16;++iteration) {
        if(2*(hi-lo)<=tolerance)return true;
        const double mid=lo+(hi-lo)*.5;
        const auto f=physical_detail::angular(I<T>(radius),I<T>(mid),p,chart);
        const auto range=physical_detail::angular(I<T>(radius),I<T>(lo,hi),p,chart);
        if(!f || !range || !range->t.valid() || !(range->t.lo>0 || range->t.hi<0))return false;
        const auto inv=I<T>(enclosure::down(1/range->t.hi),enclosure::up(1/range->t.lo));
        const auto candidate=I<T>(mid)-f->v*inv;
        double lower=double(candidate.lo),upper=double(candidate.hi);
        if(T(lower)>candidate.lo)lower=std::nextafter(lower,-INFINITY);
        if(T(upper)<candidate.hi)upper=std::nextafter(upper,INFINITY);
        const double l=std::max(lo,lower),h=std::min(hi,upper);
        if(!(h>l) || !(h-l<.8*(hi-lo)))return false;
        lo=l;hi=h;
    }
    return 2*(hi-lo)<=tolerance;
}
// Contract an already certified simple-root interval. This changes neither
// the root count nor the enclosing arithmetic; unsuccessful contraction
// leaves the incumbent Bernstein subdivision available as a fallback.
template<class T> bool contract_root(const Polynomial<T>& power,double& lo,double& hi,
                                     double angular_tolerance) {
    for(int iteration=0;iteration<12;++iteration) {
        if(2*(hi-lo)<=angular_tolerance)return true;
        const double mid=lo+(hi-lo)*.5;
        I<T> f=power[6],d=I<T>(6)*power[6],range{T(lo),T(hi)};
        for(int j=5;j>=0;--j)f=f*I<T>(mid)+power[j];
        for(int j=5;j>=1;--j)d=d*range+I<T>(j)*power[j];
        if(!d.valid() || !(d.lo>0 || d.hi<0))return false;
        const I<T> inverse(enclosure::down(T(1)/d.hi),enclosure::up(T(1)/d.lo));
        const I<T> candidate=I<T>(mid)-f*inverse;
        // Convert an enclosure outward when T is wider than binary64.
        double l=double(candidate.lo),h=double(candidate.hi);
        if(T(l)>candidate.lo)l=std::nextafter(l,-INFINITY);
        if(T(h)<candidate.hi)h=std::nextafter(h,INFINITY);
        l=std::max(lo,l);h=std::min(hi,h);
        if(!(h>l) || !(h-l<.8*(hi-lo)))return false;
        lo=l;hi=h;
    }
    return 2*(hi-lo)<=angular_tolerance;
}
}
// One immutable epoch: promoted coefficients are constructed once and reused
// at every radial node, including new nodes appended by later refinement.
struct AngularCoefficientCache {
    std::shared_ptr<point_polynomial_detail::LensContext> lens_context;
    std::shared_ptr<const point_polynomial_detail::Cache<double>> point_signs;
    std::shared_ptr<const point_polynomial_detail::Cache<long double>> wide_point_signs;
    std::size_t point_sign_requests=0,point_sign_threshold=64;
    Geometry geometry;
    std::array<double,8> identity;
    std::optional<std::array<surface_detail::RadialPolynomialT<__float128>,7>> quad;
    explicit AngularCoefficientCache(const Geometry& p):geometry(p),
        identity{p.X,p.Y,p.rho,p.m1,p.m2,p.a,p.bx,p.by}{}
    angular_detail::Polynomial<__float128> at_quad(double radius) {
        using IQ=angular_detail::I<__float128>;
        if(!quad) {
            surface_detail::RadialPolynomialT<__float128> R;R.degree=1;R.c[1]=IQ(1);
            quad=boundary_from_radius(R,geometry).coefficients;
        }
        angular_detail::Polynomial<__float128> out;
        for(int k=0;k<7;++k) {
            const auto& c=(*quad)[k];out[k]=c.c[c.degree];
            for(int j=c.degree-1;j>=0;--j)out[k]=out[k]*IQ(radius)+c.c[j];
        }
        return out;
    }
};
struct AngularBracket {double lo=0,hi=0;};
struct AngularArc {double lo=0,hi=0;};
struct ImageExclusion {Complex center;double radius=0;};
struct AngularOptions {
    double endpoint_tolerance=1e-12;
    int maximum_boxes=8192;
    bool allow_quad=true;
    bool fast_roots=true;
    const std::vector<ImageExclusion>* exclusions=nullptr;
    const ImageExclusion* inclusion=nullptr;
};
struct AngularTopology {
    std::vector<AngularBracket> roots;
    std::vector<AngularArc> arcs;
    bool completed=false;
    const char* reason="not_started";
    int double_boxes=0,quad_boxes=0,quad_promotions=0;
    // Root brackets bound the geometric angular-measure uncertainty.
    double angular_measure_error=0;
};

// -1 outside, +1 inside, 0 unresolved. An excluded ball is certified to
// contain whole image components and its flux is bounded separately.
inline int angular_membership(double R,double theta,const Geometry& p,const AngularOptions& options) {
    const Complex z=std::polar(R,theta);
    const double guard=64*std::numeric_limits<double>::epsilon()*(1+R);
    bool uncertain=false;
    if(options.inclusion) {
        const double distance=std::abs(z-options.inclusion->center);
        if(distance-guard>options.inclusion->radius)return -1;
        uncertain=distance+guard>=options.inclusion->radius;
    }
    if(options.exclusions)for(auto ball:*options.exclusions) {
        const double distance=std::abs(z-ball.center);
        if(distance+guard<ball.radius)return -1;
        uncertain=uncertain || distance-guard<=ball.radius;
    }
    double phi;
    try {phi=evaluate(R,theta,p).phi;}
    catch(const std::domain_error&){return -1;}
    if(!std::isfinite(phi) || phi==0 || (phi>0 && uncertain))return 0;
    return phi>0?1:-1;
}
inline AngularTopology finish_topology(AngularTopology out,double R,const Geometry& p,AngularOptions options={}) {
    const double pi=std::acos(-1.);
    std::sort(out.roots.begin(),out.roots.end(),[](auto x,auto y){return x.lo<y.lo;});
    if(out.roots.size()>6){out.reason="root_count_invalid";return out;}
    auto repair_bracket=[&](AngularBracket& root) {
        const double centre=(root.lo+root.hi)*.5;
        const bool chart=std::abs(std::tan(centre*.5))>1;
        double shift=chart?pi:0;
        while(centre-shift>pi)shift+=2*pi;
        while(centre-shift<-pi)shift-=2*pi;
        const double pad=32*std::numeric_limits<double>::epsilon();
        double lo=std::nextafter(std::tan((root.lo-shift)*.5)-pad,-INFINITY);
        double hi=std::nextafter(std::tan((root.hi-shift)*.5)+pad,INFINITY);
        bool ok=angular_detail::contract_physical(R,p,chart,lo,hi,1e-12);
        if(!ok && options.allow_quad) {
            ++out.quad_promotions;
            ok=angular_detail::contract_physical<__float128>(R,p,chart,lo,hi,1e-12);
            ++out.quad_boxes;
        }
        if(!ok)return false;
        root.lo=std::max(root.lo,std::nextafter(shift+2*std::atan(lo),-INFINITY));
        root.hi=std::min(root.hi,std::nextafter(shift+2*std::atan(hi),INFINITY));
        return root.hi>root.lo;
    };
    if(options.inclusion || (options.exclusions && !options.exclusions->empty())) {
        std::vector<AngularBracket> retained;
        for(auto root:out.roots) {
            const double centre=(root.lo+root.hi)*.5;
            const Complex z=std::polar(R,centre);
            const double uncertainty=R*(root.hi-root.lo)*.5+
                64*std::numeric_limits<double>::epsilon()*(1+R);
            bool excluded=false,uncertain=false;
            if(options.inclusion) {
                const double distance=std::abs(z-options.inclusion->center);
                if(distance-uncertainty>options.inclusion->radius)excluded=true;
                else uncertain=distance+uncertainty>=options.inclusion->radius;
            }
            if(options.exclusions)for(auto ball:*options.exclusions) {
                const double distance=std::abs(z-ball.center);
                if(distance+uncertainty<ball.radius){excluded=true;break;}
                uncertain=uncertain || distance-uncertainty<=ball.radius;
            }
            if(excluded)continue;
            if(uncertain){out.reason="exclusion_root_ambiguous";return out;}
            retained.push_back(root);
        }
        out.roots=std::move(retained);
    }
    for(std::size_t i=1;i<out.roots.size();++i)if(out.roots[i-1].hi>=out.roots[i].lo) {
        // Coarse flux-budget brackets can touch even though isolation proved
        // distinct roots. Tighten only this pair, preserving other roots and
        // their completeness certificate.
        if(options.endpoint_tolerance>1e-12) {
            repair_bracket(out.roots[i-1]);repair_bracket(out.roots[i]);
        }
        if(out.roots[i-1].hi>=out.roots[i].lo) {
            out.reason="overlapping_root_brackets";return out;
        }
    }
    if(out.roots.size()%2){out.reason="root_count_invalid";return out;}
    // Every accepted bracket contains one simple crossing. Membership
    // alternates, so one witness in the largest circular gap determines all
    // arcs. Avoid repeatedly evaluating tiny, cancellation-prone fold gaps.
    double witness=.123;std::size_t witness_gap=0;
    if(!out.roots.empty()) {
        double widest=out.roots.front().lo+2*pi-out.roots.back().hi;
        witness=out.roots.back().hi+widest*.5;
        for(std::size_t i=1;i<out.roots.size();++i) {
            const double width=out.roots[i].lo-out.roots[i-1].hi;
            if(width>widest){widest=width;witness=out.roots[i-1].hi+width*.5;witness_gap=i;}
        }
    }
    const int witness_sign=angular_membership(R,witness,p,options);
    if(!witness_sign){out.reason="ambiguous_arc_witness";return out;}
    const bool initial_inside=(witness_sign>0)!=(witness_gap%2!=0);
    double left=-pi*.5;
    for(std::size_t i=0;i<=out.roots.size();++i) {
        const double right=i<out.roots.size()?out.roots[i].lo:pi*1.5;
        if(!(right>left)){out.reason="chart_endpoint_root";return out;}
        if(initial_inside!=(i%2!=0)) {
            const double start=i? .5*(out.roots[i-1].lo+out.roots[i-1].hi):-pi*.5;
            const double stop=i<out.roots.size()? .5*(out.roots[i].lo+out.roots[i].hi):pi*1.5;
            out.arcs.push_back({start,stop});
        }
        if(i<out.roots.size()) {
            out.angular_measure_error+=out.roots[i].hi-out.roots[i].lo;
            left=out.roots[i].hi;
        }
    }
    // Join an arc crossing the fixed chart seam. Its two physical endpoints
    // then share one immutable angular map and one endpoint-weighted rule.
    if(out.arcs.size()>1 && out.arcs.front().lo==-pi*.5 && out.arcs.back().hi==pi*1.5) {
        AngularArc joined{out.arcs.back().lo,out.arcs.front().hi+2*pi};
        out.arcs.erase(out.arcs.begin());out.arcs.back()=joined;
    }
    out.completed=true;out.reason="completed";return out;
}
// Promote only one unresolved continued root. Previously proved siblings and
// the band's crossing count remain valid; no new global sextic isolation is
// needed merely because this root's binary64 residual is cancellation-limited.
inline std::optional<AngularBracket> continued_root_quad(double R,double coordinate,bool chart,
        const Geometry& p,const AngularOptions& options) {
    using namespace angular_detail;using Q=__float128;using IQ=I<Q>;
    Q t=coordinate;
    for(int iteration=0;iteration<12;++iteration) {
        const auto f=physical_detail::angular(Q(R),t,p,chart);
        if(!f || f->t==0 || !finiteq(f->v) || !finiteq(f->t))return std::nullopt;
        const Q step=f->v/f->t;
        if(!finiteq(step) || fabsq(step)>Q(.1))return std::nullopt;
        t-=step;
        if(fabsq(step)<Q(1e-30)*(1+fabsq(t)))break;
    }
    auto angle_bracket=[&](IQ k)->std::optional<AngularBracket> {
        auto lower=[](Q x){double d=double(x);return Q(d)>x?std::nextafter(d,-INFINITY):d;};
        auto upper=[](Q x){double d=double(x);return Q(d)<x?std::nextafter(d,INFINITY):d;};
        const double pi=std::acos(-1.);
        double lo=(chart?pi:0)+2*std::atan(lower(k.lo));
        double hi=(chart?pi:0)+2*std::atan(upper(k.hi));
        while((lo+hi)*.5<-pi*.5){lo+=2*pi;hi+=2*pi;}
        while((lo+hi)*.5>pi*1.5){lo-=2*pi;hi-=2*pi;}
        lo=std::nextafter(lo,-INFINITY);hi=std::nextafter(hi,INFINITY);
        if(hi-lo<=options.endpoint_tolerance)return AngularBracket{lo,hi};
        return std::nullopt;
    };
    const Q sign_width=Q(options.endpoint_tolerance)*Q(.15);
    const IQ bracket(enclosure::down(t-sign_width),enclosure::up(t+sign_width));
    const auto left=physical_detail::boundary_value(IQ(Q(R),Q(R)),IQ(bracket.lo,bracket.lo),p,chart);
    const auto right=physical_detail::boundary_value(IQ(Q(R),Q(R)),IQ(bracket.hi,bracket.hi),p,chart);
    if(left && right && left->sign()*right->sign()==-1) {
        auto result=angle_bracket(bracket);
        if(result)return result;
    }
    const auto point=physical_detail::angular(IQ(Q(R),Q(R)),IQ(t,t),p,chart);
    const auto f=physical_detail::angular(Q(R),t,p,chart);
    if(!point || !f || f->t==0)return std::nullopt;
    const IQ A(Q(1)/f->t,Q(1)/f->t);
    Q width=Q(options.endpoint_tolerance)*Q(.1);
    for(int attempt=0;attempt<8;++attempt) {
        const IQ domain(enclosure::down(t-width),enclosure::up(t+width));
        const auto range=physical_detail::angular(IQ(Q(R),Q(R)),domain,p,chart);
        if(range) {
            const auto residual=IQ(1)-A*range->t;
            const auto k=IQ(t,t)-A*point->v+residual*(domain-IQ(t,t));
            if(k.valid() && std::max(fabsq(residual.lo),fabsq(residual.hi))<Q(.5) &&
               k.lo>domain.lo && k.hi<domain.hi && 2*(k.hi-k.lo)<Q(options.endpoint_tolerance)*Q(.9)) {
                return angle_bracket(k);
            }
        }
        width*=4;
    }
    return std::nullopt;
}
// An event-free radial band certifies the total crossing count. Continue
// those roots locally, prove disjoint root enclosures, and account for every
// crossing; the global sextic solver remains the fallback for an unresolved
// radius. No parameter-derivative path is executed here.
inline AngularTopology continued_topology(double R,const Geometry& p,const AngularTopology& inherited,
                                         AngularOptions options={},int certified_count=-1,AngularCoefficientCache* cache=nullptr) {
    using namespace angular_detail;
    AngularTopology out;
    if((certified_count<0 && !inherited.completed) || !enclosure::environment_ok()){out.reason="continuation_unavailable";return out;}
    if(certified_count>=0 && (certified_count>6 || certified_count%2 || inherited.roots.size()!=std::size_t(certified_count))) {
        out.reason="continuation_count_witness";return out;
    }
    if(inherited.roots.empty()) {
        bool inside=!inherited.arcs.empty();
        if(certified_count==0) {
            const int sign=angular_membership(R,.123,p,options);
            if(!sign){out.reason="continuation_zero_witness";return out;}
            inside=sign>0;
        }
        if(inside)out.arcs={{-std::acos(-1.)*.5,std::acos(-1.)*1.5}};
        out.completed=true;out.reason="continued_zero_count";return out;
    }
    const double pi=std::acos(-1.),eps=std::numeric_limits<double>::epsilon();
    const physical_detail::Ray<double> ray(R,p);
    const physical_detail::Ray<I<double>> proof_ray(I<double>(R),p);
    std::optional<point_polynomial_detail::Cache<double>::Radius> point_polynomial;
    std::optional<point_polynomial_detail::Cache<long double>::Radius> wide_point_polynomial;
    if(cache && cache->identity==std::array<double,8>{p.X,p.Y,p.rho,p.m1,p.m2,p.a,p.bx,p.by} &&
       ++cache->point_sign_requests>=cache->point_sign_threshold) {
        // Polynomial signs imply physical signs only away from lens poles.
        // Verify absence of poles on this whole radius circle once.
        const auto r2=I<double>(R)*I<double>(R);
        if((r2-I<double>(p.a)*I<double>(p.a)).sign()!=0 &&
           (r2-I<double>(p.bx)*I<double>(p.bx)-I<double>(p.by)*I<double>(p.by)).sign()!=0) {
            if(options.endpoint_tolerance>=1e-10) {
                if(!cache->point_signs)cache->point_signs=std::make_shared<const point_polynomial_detail::Cache<double>>(cache->lens_context?cache->lens_context->table<double>(p):point_polynomial_detail::Cache<double>(p));
                point_polynomial=cache->point_signs->at(R);
            } else if constexpr(std::numeric_limits<long double>::digits==64) {
                if(!cache->wide_point_signs)cache->wide_point_signs=std::make_shared<const point_polynomial_detail::Cache<long double>>(cache->lens_context?cache->lens_context->table<long double>(p):point_polynomial_detail::Cache<long double>(p));
                wide_point_polynomial=cache->wide_point_signs->at(R);
            }
        }
    }
    for(auto root:inherited.roots) {
        const double angle=root.lo+(root.hi-root.lo)*.5;
        double coordinate=std::tan(angle*.5);bool chart=std::abs(coordinate)>1;
        if(chart)coordinate=-1/coordinate;
        bool solved=false;
        // The accepted line-search evaluation is the next Newton centre.
        // Retain it instead of evaluating the same point twice.
        auto f=ray.angular(coordinate,chart);
        for(int iteration=0;iteration<24;++iteration) {
            if(!f || f->t==0 || !std::isfinite(f->v) || !std::isfinite(f->t))break;
            double step=f->v/f->t;
            if(!std::isfinite(step))break;
            // Keep the accepted centre when its next correction is already
            // below the endpoint budget. The interval proof below decides
            // acceptance; another rounded Newton evaluation adds no witness.
            if(std::abs(step)<std::max(16*eps*(1+std::abs(coordinate)),options.endpoint_tolerance*.02))break;
            if(std::abs(step)>1)step=std::copysign(1.,step);
            double candidate=coordinate-step;
            bool accepted=false;
            std::optional<physical_detail::AngularEvaluation<double>> next;
            for(int retry=0;retry<12;++retry) {
                if(std::abs(candidate)<4) {
                    next=ray.angular(candidate,chart);
                    if(next && (std::abs(next->v)<std::abs(f->v) ||
                       std::abs(step)<16*eps*(1+std::abs(coordinate)))){accepted=true;break;}
                }
                step*=.5;candidate=coordinate-step;
            }
            if(!accepted)break;
            coordinate=candidate;f=next;
            if(std::abs(step)<std::max(16*eps*(1+std::abs(coordinate)),options.endpoint_tolerance*.02)){solved=true;break;}
        }
        // A represented centre need not meet a scalar residual threshold:
        // acceptance depends only on the following interval inclusion proof.
        (void)solved;
        // The band already certifies the total number of roots. Disjoint
        // sign-changing brackets give that many existing roots, hence also
        // uniqueness/completeness, without a Jacobian enclosure per root.
        const double sign_width=std::max(options.endpoint_tolerance*.2,64*eps*(1+std::abs(coordinate)));
        const double left=std::nextafter(coordinate-sign_width,-INFINITY),right=std::nextafter(coordinate+sign_width,INFINITY);
        auto sign_at=[&](double t) {
            if(point_polynomial){const int sign=point_polynomial->sign(t,chart);if(sign)return sign;}
            if(wide_point_polynomial){const int sign=wide_point_polynomial->sign(t,chart);if(sign)return sign;}
            const auto value=proof_ray.value(I<double>(t),chart);
            return value?value->sign():0;
        };
        const int left_sign=sign_at(left),right_sign=sign_at(right);
        if(left_sign*right_sign==-1 && 2*(right-left)<=options.endpoint_tolerance) {
            double lo=(chart?pi:0)+2*std::atan(left),hi=(chart?pi:0)+2*std::atan(right);
            while((lo+hi)*.5<-pi*.5){lo+=2*pi;hi+=2*pi;}
            while((lo+hi)*.5>pi*1.5){lo-=2*pi;hi-=2*pi;}
            out.roots.push_back({std::nextafter(lo,-INFINITY),std::nextafter(hi,INFINITY)});
            ++out.double_boxes;continue;
        }
        auto centre=proof_ray.angular(I<double>(coordinate),chart);
        const auto& estimate=f;
        if(!centre || !estimate || estimate->t==0 || !std::isfinite(estimate->t)) {
            out.reason="continuation_centre";return out;
        }
        const I<double> A(1/estimate->t);
        double width=std::max(options.endpoint_tolerance*.25,64*eps*(1+std::abs(coordinate)));
        bool proven=false;
        const bool residual_limited=std::abs(A.lo)*(centre->v.hi-centre->v.lo)>options.endpoint_tolerance*.5;
        for(int attempt=0;attempt<(residual_limited?0:12);++attempt) {
            ++out.double_boxes;
            const I<double> domain(std::nextafter(coordinate-width,-INFINITY),std::nextafter(coordinate+width,INFINITY));
            auto range=proof_ray.angular(domain,chart);
            if(range) {
                const auto residual=I<double>(1)-A*range->t;
                const auto k=I<double>(coordinate)-A*centre->v+residual*(domain-I<double>(coordinate));
                if(k.valid() && std::max(std::abs(residual.lo),std::abs(residual.hi))<.5 &&
                   k.lo>domain.lo && k.hi<domain.hi && 2*(k.hi-k.lo)<=options.endpoint_tolerance) {
                    double lo=(chart?pi:0)+2*std::atan(k.lo),hi=(chart?pi:0)+2*std::atan(k.hi);
                    while((lo+hi)*.5<-pi*.5){lo+=2*pi;hi+=2*pi;}
                    while((lo+hi)*.5>pi*1.5){lo-=2*pi;hi-=2*pi;}
                    out.roots.push_back({std::nextafter(lo,-INFINITY),std::nextafter(hi,INFINITY)});
                    proven=true;break;
                }
            }
            width*=4;
        }
        if(!proven && options.allow_quad) {
            ++out.quad_promotions;
            auto bracket=continued_root_quad(R,coordinate,chart,p,options);
            if(bracket){out.roots.push_back(*bracket);++out.quad_boxes;proven=true;}
        }
        if(!proven){out.reason="continuation_inclusion";return out;}
    }
    if(out.roots.size()!=inherited.roots.size()){out.reason="continuation_count";return out;}
    auto finished=finish_topology(std::move(out),R,p,options);
    if(finished.completed && finished.roots.size()!=inherited.roots.size()) {
        finished.completed=false;finished.reason="continuation_count";
    }
    return finished;
}
inline AngularTopology angular_topology(double R,const Geometry& p,
                                        AngularOptions options={},AngularCoefficientCache* cache=nullptr) {
    using namespace angular_detail;
    AngularTopology out;
    p.validate();
    if(cache && cache->identity!=std::array<double,8>{p.X,p.Y,p.rho,p.m1,p.m2,p.a,p.bx,p.by}) {
        out.reason="coefficient_cache_identity";return out;
    }
    if(!(R>0) || !std::isfinite(R) || !(options.endpoint_tolerance>0) ||
       !std::isfinite(options.endpoint_tolerance) || options.maximum_boxes<1) {
        out.reason="invalid_config";return out;
    }
    if(!enclosure::environment_ok()) {out.reason="floating_environment";return out;}
    const auto coefficients=boundary_as<I<double>>(R,p).coefficients;
    std::optional<Polynomial<__float128>> quad_coefficients;
    const double pi=std::acos(-1.0);
    bool failed=false;
    auto angle=[&](double t,bool reciprocal_chart) {
        return (reciprocal_chart?pi:0)+2*std::atan(t);
    };
    auto accept=[&](double lo,double hi,bool reciprocal_chart) {
        // libm conversion is not an interval operation. Expand the reported
        // angular bracket; polynomial sign decisions remain in the charts.
        out.roots.push_back({std::nextafter(angle(lo,reciprocal_chart),-INFINITY),
                             std::nextafter(angle(hi,reciprocal_chart),INFINITY)});
    };
    // Refine only a promoted box. Previously excluded boxes and accepted roots
    // survive escalation; quad coefficients are lazily reconstructed once.
    auto isolate_quad=[&](double lo,double hi,bool reciprocal_chart) {
        if(!options.allow_quad){out.reason="precision_required";failed=true;return;}
        ++out.quad_promotions;
        if(!quad_coefficients)
            quad_coefficients=options.fast_roots && cache?cache->at_quad(R):boundary_as<I<__float128>>(R,p).coefficients;
        std::vector<std::pair<double,double>> contracted_roots;int contracted_work=0;
        if(options.fast_roots && monotone_roots(chart(*quad_coefficients,reciprocal_chart),6,lo,hi,
                options.endpoint_tolerance*.5,contracted_roots,contracted_work,
                options.maximum_boxes-out.double_boxes-out.quad_boxes)) {
            out.quad_boxes+=contracted_work;
            for(auto root:contracted_roots)accept(root.first,root.second,reciprocal_chart);
            return;
        }
        struct Box {double lo,hi;Polynomial<__float128> b;};
        std::vector<Box> pending{{lo,hi,restrict(chart(*quad_coefficients,reciprocal_chart),lo,hi)}};
        while(!pending.empty() && !failed) {
            auto box=pending.back();pending.pop_back();++out.quad_boxes;
            if(out.double_boxes+out.quad_boxes>options.maximum_boxes) {
                out.reason="box_budget";failed=true;break;
            }
            const int n=variations(box.b);
            // A zero endpoint may be an exact shared root. Do not silently drop
            // it or count it twice. The event layer must resolve that case.
            if(box.b[0].sign()==0 || box.b[6].sign()==0) {
                out.reason="multiple_or_chart_endpoint_root";failed=true;break;
            }
            if(n==0)continue;
            if(options.fast_roots && n==1 && box.b[0].sign()*box.b[6].sign()==-1) {
                double l=box.lo,h=box.hi;
                if(contract_physical(R,p,reciprocal_chart,l,h,options.endpoint_tolerance) ||
                   contract_root(chart(*quad_coefficients,reciprocal_chart),l,h,options.endpoint_tolerance)) {
                    accept(l,h,reciprocal_chart);continue;
                }
            }
            if(n==1 && box.b[0].sign()*box.b[6].sign()==-1 &&
               angle(box.hi,reciprocal_chart)-angle(box.lo,reciprocal_chart)<=options.endpoint_tolerance) {
                accept(box.lo,box.hi,reciprocal_chart);continue;
            }
            const double mid=box.lo+(box.hi-box.lo)*.5;
            if(!(mid>box.lo && mid<box.hi)) {
                out.reason="unresolved_root_cluster";failed=true;break;
            }
            auto halves=split(box.b);
            pending.push_back({mid,box.hi,halves.second});
            pending.push_back({box.lo,mid,halves.first});
        }
    };
    for(bool reciprocal_chart:{false,true}) {
        const auto power=chart(coefficients,reciprocal_chart);
        std::vector<std::pair<double,double>> fast_roots;int fast_work=0;
        if(options.fast_roots && monotone_roots(power,6,-1.,1.,std::max(options.endpoint_tolerance*.5,1e-7),fast_roots,fast_work,options.maximum_boxes)) {
            out.double_boxes+=fast_work;
            for(auto root:fast_roots) {
                double lo=root.first,hi=root.second;
                if(contract_physical(R,p,reciprocal_chart,lo,hi,options.endpoint_tolerance))accept(lo,hi,reciprocal_chart);
                else isolate_quad(lo,hi,reciprocal_chart);
            }
            if(failed)break;
            continue;
        }
        struct Box {double lo,hi;Polynomial<double> b;};
        std::vector<Box> pending{{-1,1,restrict(chart(coefficients,reciprocal_chart),-1,1)}};
        while(!pending.empty() && !failed) {
            auto box=pending.back();pending.pop_back();++out.double_boxes;
            if(out.double_boxes+out.quad_boxes>options.maximum_boxes) {
                out.reason="box_budget";failed=true;break;
            }
            const int n=variations(box.b);
            if(n<0 || box.b[0].sign()==0 || box.b[6].sign()==0) {
                isolate_quad(box.lo,box.hi,reciprocal_chart);continue;
            }
            if(n==0)continue;
            if(options.fast_roots && n==1 && box.b[0].sign()*box.b[6].sign()==-1) {
                double lo=box.lo,hi=box.hi;
                if(contract_physical(R,p,reciprocal_chart,lo,hi,options.endpoint_tolerance) ||
                   contract_root(power,lo,hi,options.endpoint_tolerance)) {
                    accept(lo,hi,reciprocal_chart);continue;
                }
            }
            if(n==1 && box.b[0].sign()*box.b[6].sign()==-1 &&
               angle(box.hi,reciprocal_chart)-angle(box.lo,reciprocal_chart)<=options.endpoint_tolerance) {
                accept(box.lo,box.hi,reciprocal_chart);continue;
            }
            const double mid=box.lo+(box.hi-box.lo)*.5;
            if(!(mid>box.lo && mid<box.hi)) {
                isolate_quad(box.lo,box.hi,reciprocal_chart);continue;
            }
            auto halves=split(box.b);
            pending.push_back({mid,box.hi,halves.second});
            pending.push_back({box.lo,mid,halves.first});
        }
        if(failed)break;
    }
    if(failed)return out;
    return finish_topology(std::move(out),R,p,options);
}
} // namespace lcbinint::algebraic_boundary::triple
#if defined(__GNUC__)
#pragma GCC pop_options
#endif
