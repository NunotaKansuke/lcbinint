#include "experiments/boundary-support-overlay/src/lcbinint/magnification/bennett_integration.hpp"
#include "experiments/boundary-support-overlay/src/lcbinint/magnification/cartesian_run_fill.hpp"
#include <cassert>
#include <limits>
int main() {
 using namespace lcbinint::magnification::detail;
 constexpr long double pi=3.1415926535897932384626433832795L;
 // Independent finite-interval quadrature of exactly parabolic and cubic
 // source depths. Transformation removes the endpoint square roots.
 for(double left:{.1,.4,.9})for(double right:{.15,.5,.95})
 for(double c:{0.,.2,.5,.8,1.})for(double cubic:{-.02,0.,.02}) {
  constexpr double k=.3;const double depth=k*left*right;
  const auto m=finite_one_node_ld_model(left,right,depth,c,6*std::abs(cubic));assert(m.resolved);
  long double sum=0;constexpr int n=1024;const long double half=(left+right)/2,mid=(right-left)/2;
  for(int j=0;j<=n;++j) {
   const long double theta=pi*j/n,sine=std::sin(theta),t=mid+half*std::cos(theta);
   const long double f=((1-c)+c*half*sine*std::sqrt(k+cubic*t))*half*sine;
   sum+=(j==0||j==n?1:j%2?4:2)*f;
  }
  const long double exact=sum*pi/(3*n);
  assert(std::abs(exact-m.integral)<=m.uncertainty+1e-11L);
  if(cubic==0)assert(std::abs(exact-m.integral)<1e-11L);
 }
 // Closed gaps with overlapping endpoint stencils must use a finite
 // model or local curvature fallback, never two independent infinite tails.
 const auto image=[](const std::array<double,6>& widths) {
  std::vector<SameGridErrorNode> nodes{{-1,-10,10,20,1,20,0}};
  for(int y=0;y<6;++y) {
   const double half=widths[y]/2;
   nodes.push_back({y,-10,-half,10-half,1,10-half,0});
   nodes.push_back({y,half,10,10-half,1,10-half,0});
  }
  nodes.push_back({6,-10,10,20,1,20,0});return nodes;
 };
 std::array<double,6> exact_widths{};double used=0;
 for(int j=0;j<6;++j){exact_widths[j]=std::sqrt(2.8*2.8-(j-2.5)*(j-2.5));used+=exact_widths[j];}
 const auto exact_gap=same_grid_internal_error(image(exact_widths),1,true);
 assert(exact_gap.resolved);
 assert(std::abs(exact_gap.signed_residual+(double(pi)*2.8*2.8/2-used))<1e-10);
 const std::array<double,6> bad_widths{.6,.8,1.,1.2,1.4,1.6};
 const auto invalid_gap=same_grid_internal_error(image(bad_widths),1,true);
 double curvature=0;
 for(int j=0;j<6;++j)curvature+=std::abs((j?bad_widths[j-1]:0)-2*bad_widths[j]+(j<5?bad_widths[j+1]:0));
 assert(!invalid_gap.resolved && invalid_gap.uncertainty>=curvature/24);
 assert(invalid_gap.signed_residual==0);
 assert(!finite_one_node_ld_model(0,.2,.1,.5,0).resolved);
 assert(!finite_one_node_ld_model(.5,.5,.01,1,100).resolved);
 assert(!finite_one_node_ld_model(.5,.5,.1,1,std::numeric_limits<double>::infinity()).resolved);
 // Bernoulli defects checked against independently summed finite lattices
 // with the exact polynomial midpoint tail, including new powers3 and4.
 for(double d:{.01,.15,.6,1.,1.49})for(double w0:{.2,1.,1.3})for(double w1:{.5,1.,1.4})
 for(int p=1;p<=4;++p) {
  const int count=8;const long double end=d+count-.5L;
  long double exact=std::pow(end,p+1)/(p+1);
  for(int j=0;j<count;++j)exact-=(j==0?w0:j==1?w1:1)*std::pow((long double)d+j,p);
  exact-=p*std::pow(end,p-1)/24;
  if(p>=3)exact+=7*p*(p-1)*(p-2)*std::pow(end,p-3)/5760;
  assert(std::abs(exact-integer_endpoint_quadrature_defect(d,w0,w1,p))<1e-9L);
 }
 for(double d:{.01,.15,.6,1.,1.49})for(double w0:{.2,1.,1.3})for(double w1:{.5,1.,1.4}) {
  const long double exact=power_lattice_endpoint_constant(d,3)-(w0-1)*std::pow((long double)d,3.5L)-(w1-1)*std::pow((long double)d+1,3.5L);
  assert(std::abs(exact-power_endpoint_quadrature_defect(d,w0,w1,3))<2e-8L);
 }
}
