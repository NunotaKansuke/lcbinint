#include <cmath>
#include <iostream>
#include "lcbinint/magnification/holonomic/adaptive_radial.hpp"
using namespace lcbinint::holonomic;
int main(){int checks=0,failures=0;double maximum=0;
for(int level=3;level<=8;++level){const int m=1<<level,M=m/2;const auto& rule=fejer_rule(level);
for(int mode=-2;mode<=M-2;++mode){std::array<double,128> values{};std::array<double,256> predicted{};double magnitude=1;
for(int j=1;j<M;++j){const double theta=3.14159265358979323846*j/M,x=fejer_node(level-1,j);values[j]=mode==-2?std::exp(x):mode==-1?std::sin(71*x)+std::cos(13*x):std::cos(mode*theta);magnitude=std::max(magnitude,std::fabs(values[j]));}
prediction_detail::sine_predict(m,values.data(),predicted.data());
for(int k=1;k<m;k+=2){adaptive_detail::Sum dense;for(int j=1;j<M;++j)dense.add(rule.interp[(k/2)*(M-1)+j-1]*values[j]);double err=std::fabs(predicted[k]-dense.get())/magnitude;maximum=std::max(maximum,err);++checks;if(!(err<=2e-11)){++failures;std::cerr<<level<<' '<<mode<<' '<<k<<' '<<err<<'\n';}
if(mode>=0){const double exact=std::cos(mode*3.14159265358979323846*k/m);++checks;if(!(std::fabs(predicted[k]-exact)<=2e-11))++failures;}}
}}
std::cout<<checks<<" checks "<<failures<<" failures; maximum relative-to-scale dense discrepancy "<<maximum<<'\n';return failures?1:0;}
