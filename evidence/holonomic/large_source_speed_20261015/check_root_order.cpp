#include <cmath>
#include <limits>
#include <random>
#include <cstring>
#include <iostream>
#include <cfenv>
int main(){
 auto root_less=[](double a,double b,bool inclusive){const double maximum=std::max(a,b);
#if !defined(__FAST_MATH__)
 if(std::isfinite(a)&&std::isfinite(b)&&a>=0&&b>=0&&maximum>=std::numeric_limits<double>::min()&&std::fabs(a-b)>8*std::numeric_limits<double>::epsilon()*maximum)return inclusive?a<=b:a<b;
#endif
 return inclusive?std::sqrt(a)<=std::sqrt(b):std::sqrt(a)<std::sqrt(b);};
 std::mt19937_64 rng(80924);int checks=0;
 for(int mode:{FE_TONEAREST,FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO}){std::fesetround(mode);
 for(int k=0;k<200000;++k){auto bits=rng();double a;std::memcpy(&a,&bits,8);bits=rng();double b;std::memcpy(&b,&bits,8);if(k&1){b=a;for(unsigned n=0;n<rng()%24;++n)b=std::nextafter(b,INFINITY);}
 for(bool inclusive:{false,true}){bool expected=inclusive?std::sqrt(a)<=std::sqrt(b):std::sqrt(a)<std::sqrt(b);if(root_less(a,b,inclusive)!=expected){std::cerr<<"mismatch\\n";return 1;}++checks;}}}
 std::fesetround(FE_TONEAREST);std::cout<<checks<<" sqrt-order comparisons agree across four IEEE rounding modes\\n";}
