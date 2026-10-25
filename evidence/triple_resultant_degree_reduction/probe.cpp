#include "lcbinint/magnification/algebraic_boundary/triple/support_resultant.hpp"
#include "support-before.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
using namespace lcbinint::algebraic_boundary::triple;
using Clock=std::chrono::steady_clock;
volatile long double sink;
int main(){int matrix_checks=0,direct_checks=0,warm_checks=0;std::vector<double> before,after;
 for(int k=0;k<64;++k){Geometry p{.1+.001*k,.15-.0007*k,.03+.002*k,.2,.08,1.,-.4,.6};if(k%3==0){p.by=0;p.bx=-.7;}if(k%5==0){p.m1=.001;p.m2=.0001;}
 auto old=[&](){auto start=Clock::now();resultant_before::MatrixBasis b(p);sink=b.coefficients[3][0][0][0].value;return std::pair{b,std::chrono::duration<double>(Clock::now()-start).count()};};
 auto now=[&](){auto start=Clock::now();resultant_detail::MatrixBasis b(p);sink=b.coefficients[3][0][0][0].value;return std::pair{b,std::chrono::duration<double>(Clock::now()-start).count()};};
 auto a=now();auto b=old();if(k%2){b=old();a=now();}else{a=now();b=old();}before.push_back(b.second);after.push_back(a.second);
 for(int h=0;h<10;++h)for(int i=0;i<5;++i)for(int j=i;j<5;++j)for(int n=0;n<9;++n){auto x=a.first.coefficients[h][i][j][n],y=b.first.coefficients[h][i][j][n];if(x.interval().hi<y.interval().lo || x.interval().lo>y.interval().hi){std::cerr<<"matrix mismatch "<<k<<" "<<h<<" "<<i<<" "<<j<<" "<<n<<"\n";return 1;}++matrix_checks;}
 auto x=resultant_detail::support_raw_direct(p,.04,3.1),y=resultant_before::support_raw_direct(p,.04,3.1);
 for(int j=0;j<29;++j){if(x[j].value!=y[j].value||x[j].error!=y[j].error)return 2;++direct_checks;}
 for(int w=0;w<6;++w){x=resultant_detail::support_raw(p,.04,3.1);y=resultant_before::support_raw(p,.04,3.1);}
 for(int j=0;j<29;++j){if(x[j].interval().hi<y[j].interval().lo||x[j].interval().lo>y[j].interval().hi)return 3;++warm_checks;}
 }
 std::vector<double> ratios;for(int i=0;i<64;++i)ratios.push_back(before[i]/after[i]);std::sort(ratios.begin(),ratios.end());std::sort(before.begin(),before.end());std::sort(after.begin(),after.end());
 std::cout<<"{\"matrix_checks\":"<<matrix_checks<<",\"direct_checks\":"<<direct_checks<<",\"warm_checks\":"<<warm_checks<<",\"before_ms\":"<<before[32]*1000<<",\"after_ms\":"<<after[32]*1000<<",\"median_preparation_speedup\":"<<ratios[32]<<"}\n";
}
