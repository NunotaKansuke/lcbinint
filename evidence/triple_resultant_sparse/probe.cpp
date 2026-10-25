#include "lcbinint/magnification/algebraic_boundary/triple/support_resultant.hpp"
#include "support-before.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace lcbinint::algebraic_boundary::triple;
volatile long double sink;
int main(){int checks=0;std::vector<double> old_times,new_times;
 for(int k=0;k<120;++k){Geometry p{.1+.001*k,.15-.0007*k,.03+.002*k,.2,.08,1.,-.4,.6};
 if(k%3==0){p.by=0;p.bx=-.7;}if(k%5==0){p.m1=.001;p.m2=.0001;}
 for(int warm=0;warm<6;++warm)(void)resultant_detail::support_raw(p,.04,3.1);
 auto a=resultant_detail::support_raw(p,.04,3.1),b=resultant_before::support_raw(p,.04,3.1);
 for(int j=0;j<29;++j){if(a[j].interval().hi<b[j].interval().lo || a[j].interval().lo>b[j].interval().hi){std::cerr<<k<<" "<<j<<" mismatch\n";return 1;}++checks;}
 auto time=[&](bool old){auto t=std::chrono::steady_clock::now();for(int j=0;j<8;++j){auto c=old?resultant_before::support_raw(p,.04,3.1):resultant_detail::support_raw(p,.04,3.1);sink=c[j].value;}return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/8;};
 double x,y;if(k%2){x=time(true);y=time(false);}else{y=time(false);x=time(true);}old_times.push_back(x);new_times.push_back(y);
 }
 std::vector<double> ratios;for(int i=0;i<120;++i)ratios.push_back(old_times[i]/new_times[i]);std::sort(ratios.begin(),ratios.end());
 std::cout<<"{\"overlap_checks\":"<<checks<<",\"median_coefficient_speedup\":"<<ratios[60]<<"}\n";
}
