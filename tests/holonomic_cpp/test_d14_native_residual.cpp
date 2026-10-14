#include "lcbinint/magnification/holonomic/radial_events.hpp"
#include <random>
#include <cstdio>
#include <cstdlib>
using namespace lcbinint::holonomic;
void check(bool b){if(!b){std::fputs("native residual check failed\n",stderr);std::exit(1);}}
int main(){
 std::mt19937_64 gen(1717);int checks=0;
 for(int trial=0;trial<200;++trial){
  std::array<__float128,15> c{};__float128 scale=0;
  for(auto& x:c){x=(__float128)(int(gen()%201)-100)/32+(__float128)1e-30;scale=fmaxq(scale,fabsq(x));}
  std::vector<Cplx<__float128>> roots(14);
  for(auto& z:roots){z.re=(__float128)(int(gen()%201)-100)/32+(__float128)1e-30;z.im=trial%2?0:(__float128)(int(gen()%201)-100)/64;}
  auto bound=native_residual_detail::evaluate(c.data(),14,roots,scale);check(bound.valid);
  for(size_t i=0;i<roots.size();++i){auto q=cabs(poly_eval_c(c.data(),14,roots[i]))/(scale+(__float128)1e-300);check((__float128)bound.root_upper[i]>=q);++checks;}
  bool bounded=false;
  auto screened=re_detail::d14_screened_residual(c.data(),14,roots,scale,bounded);
  check(screened>=re_detail::d14_worst_res(c.data(),14,roots,scale));
 }
 std::array<__float128,15> c{};c[0]=1;c[14]=-1;
 std::vector<Cplx<__float128>> roots(14,Cplx<__float128>(1,0));
 auto bound=native_residual_detail::evaluate(c.data(),14,roots,1);check(bound.valid&&bound.upper<1e-13L);
 // Deliberately duplicate roots: these tests check residual only, not solve
 // success. Global completeness remains the independent solver gate.
 bool bounded=false;auto screened=re_detail::d14_screened_residual(c.data(),14,roots,1,bounded);
 check(bounded&&screened<=1e-13Q);
 roots[0].re=2;
 screened=re_detail::d14_screened_residual(c.data(),14,roots,1,bounded);
 check(bounded&&screened==re_detail::d14_worst_res(c.data(),14,roots,1));
 roots[0].re=1;
 c[0]=(__float128)1e-250;check(!native_residual_detail::evaluate(c.data(),14,roots,1).valid);c[0]=1;
 roots.assign(14,Cplx<__float128>((__float128)1e-178,0));c[14]=0;
 check(!native_residual_detail::evaluate(c.data(),14,roots,1).valid);
 c[14]=-1;roots.assign(14,Cplx<__float128>(1,0));
 std::fesetround(FE_UPWARD);check(!native_residual_detail::evaluate(c.data(),14,roots,1).valid);std::fesetround(FE_TONEAREST);
#if defined(__i386__) || defined(__x86_64__)
 unsigned short control;__asm__ volatile("fnstcw %0":"=m"(control));
 unsigned short reduced=(control&~0x0300)|0x0200;__asm__ volatile("fldcw %0"::"m"(reduced));
 check(!native_residual_detail::evaluate(c.data(),14,roots,1).valid);
 __asm__ volatile("fldcw %0"::"m"(control));
#endif
 std::printf("native residual PASS; qf comparisons %d\n",checks);
}
