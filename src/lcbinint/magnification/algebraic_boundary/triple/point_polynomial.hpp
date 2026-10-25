#pragma once
#include "geometry.hpp"
#include "coefficient_ball.hpp"
#include <atomic>
#include <memory>
#include <mutex>
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math","fp-contract=off")
#endif
namespace lcbinint::algebraic_boundary::triple::point_polynomial_detail {
template<class T> struct Poly {
 using B=coefficient_ball_detail::Ball<T>;std::array<B,9> c{};int degree=0;
 Poly()=default;Poly(double x){c[0]=B(x);}
};
template<class T> Poly<T> operator-(Poly<T>a){for(auto&x:a.c)x=-x;return a;}
template<class T> Poly<T> operator+(Poly<T>a,const Poly<T>&b){a.degree=std::max(a.degree,b.degree);for(int i=0;i<=b.degree;++i)a.c[i]=a.c[i]+b.c[i];return a;}
template<class T> Poly<T> operator-(Poly<T>a,const Poly<T>&b){return a+(-b);}
template<class T> Poly<T> operator*(const Poly<T>&a,const Poly<T>&b){Poly<T> r;r.degree=a.degree+b.degree;if(r.degree>8)throw std::logic_error("radial degree");for(int i=0;i<=a.degree;++i)for(int j=0;j<=b.degree;++j)r.c[i+j]=r.c[i+j]+a.c[i]*b.c[j];return r;}
template<class T> struct Cache {
 using B=coefficient_ball_detail::Ball<T>;
 std::array<Poly<T>,7> coefficients;
 Cache()=default;
 Cache(const Geometry&p){Poly<T> R;R.degree=1;R.c[1]=B(1);coefficients=boundary_from_radius(R,p).coefficients;}
 struct Radius {
  std::array<B,7> c;
  int sign(double x,bool chart)const {
   B f=chart?c[0]:c[6];const B t(x);
   for(int i=5;i>=0;--i){B a=chart?c[6-i]:c[i];if(chart && i%2)a=-a;f=f*t+a;}
   if(!std::isfinite(f.value)||!std::isfinite(f.error)||f.error<0)return 0;
   return f.value>f.error?1:f.value<-f.error?-1:0;
  }
 };
 Radius at(double r)const{Radius out;for(int k=0;k<7;++k){const auto&p=coefficients[k];B a=p.c[p.degree];for(int j=p.degree-1;j>=0;--j)a=a*B(r)+p.c[j];out.c[k]=a;}return out;}
};
template<class T> struct LensCache {
 using P=Poly<T>;using B=coefficient_ball_detail::Ball<T>;using C=coefficient_detail::C<P>;
 std::array<double,5> identity;
 std::array<std::array<P,7>,4> basis{};
 static std::array<P,7> transform(const std::array<C,4>& h) {
  std::array<P,7> result{};
  for(int k=0;k<4;++k) {
   std::array<C,7> poly{};poly[0]=C{P(1)};int degree=0;
   for(int i=0;i<2*k;++i){std::array<C,7> next{};for(int j=0;j<=degree;++j){next[j]=next[j]+poly[j];next[j+1]=next[j+1]+C{P(0),P(1)}*poly[j];}poly=next;++degree;}
   for(int i=0;i<3-k;++i){std::array<C,7> next{};for(int j=0;j<=degree;++j){next[j]=next[j]+poly[j];next[j+2]=next[j+2]+poly[j];}poly=next;degree+=2;}
   for(int j=0;j<7;++j)result[j]=result[j]+P(k?2:1)*(h[k]*poly[j]).re;
  }
  return result;
 }
 explicit LensCache(const Geometry&p):identity{p.m1,p.m2,p.a,p.bx,p.by} {
  auto lens=p;lens.X=lens.Y=lens.rho=0;P R;R.degree=1;R.c[1]=B(1);const auto b=boundary_from_radius(R,lens);
  const auto& n=b.numerator;const auto& d=b.denominator;std::array<C,4> base{},area{},crossx{},crossy{};
  for(int k=0;k<4;++k) {
   C a{},c{};
   for(int j=0;j+k<4;++j)base[k]=base[k]-n[j+k]*coefficient_detail::conj(n[j]);
   for(int j=0;j+k<3;++j)area[k]=area[k]+d[j+k]*coefficient_detail::conj(d[j]);
   for(int j=0;j<3 && j+k<4;++j)a=a+n[j+k]*coefficient_detail::conj(d[j]);
   for(int j=0;j<4 && j+k<3;++j)c=c+d[j+k]*coefficient_detail::conj(n[j]);
   crossx[k]=a+c;crossy[k]=C{P(0),P(1)}*(c-a);
  }
  basis={transform(base),transform(area),transform(crossx),transform(crossy)};
 }
 Cache<T> query(const Geometry&p)const {
  if(identity!=std::array<double,5>{p.m1,p.m2,p.a,p.bx,p.by})throw std::invalid_argument("lens coefficient identity");
  Cache<T> result;const B x(p.X),y(p.Y),area=B(p.rho)*B(p.rho)-x*x-y*y;
  for(int k=0;k<7;++k) {
   auto& out=result.coefficients[k];out.degree=std::max({basis[0][k].degree,basis[1][k].degree,basis[2][k].degree,basis[3][k].degree});
   for(int i=0;i<=out.degree;++i)out.c[i]=basis[0][k].c[i]+basis[1][k].c[i]*area+basis[2][k].c[i]*x+basis[3][k].c[i]*y;
  }
  return result;
 }
};
// Geometry-owned, immutable basis data. Source-dependent tables remain
// private to an integration. Plans may safely share the lazy lens basis.
class LensContext {
    std::array<double,5> identity_;
    std::atomic<std::size_t> queries_{0};
    std::once_flag wide_once_;
    std::unique_ptr<const LensCache<long double>> wide_basis_;
public:
    explicit LensContext(const Geometry&p):identity_{p.m1,p.m2,p.a,p.bx,p.by}{}
    bool matches(const Geometry&p)const{return identity_==std::array<double,5>{p.m1,p.m2,p.a,p.bx,p.by};}
    void note_query(){queries_.fetch_add(1,std::memory_order_relaxed);}
    bool reuse_ready()const{return queries_.load(std::memory_order_relaxed)>=4;}
    template<class T> Cache<T> table(const Geometry&p) {
        if(!matches(p))throw std::invalid_argument("lens context identity");
        if constexpr(std::is_same_v<T,double>) {
            // Source-basis binary64 enclosures are wider. Its integrated
            // trial was slower, so retain direct construction at this tier.
            return Cache<T>(p);
        } else {
            if(!reuse_ready())return Cache<T>(p);
            static_assert(std::is_same_v<T,long double>,"unsupported point-sign precision");
            std::call_once(wide_once_,[&]{wide_basis_=std::make_unique<const LensCache<long double>>(p);});
            return wide_basis_->query(p);
        }
    }
};

}

#if defined(__GNUC__)
#pragma GCC pop_options
#endif
