#include <chrono>
#include <iostream>
#include <iomanip>
#include "lcbinint/magnification/holonomic/adaptive_radial.hpp"
#include "lcbinint/magnification/holonomic/chebyshev_prediction.hpp"
using namespace lcbinint::holonomic;
int main(){std::cout<<std::setprecision(17)<<"level nodes direct_ms gather_ms dst_ms max_abs_prediction_difference\n";volatile double sink=0;
for(int level=3;level<=8;++level){int m=1<<level,M=m/2;const auto& r=fejer_rule(level);std::array<double,256> values{},predicted{};std::array<AdaptiveSample,128> samples{};
for(int j=1;j<M;++j){double x=fejer_node(level-1,j);values[j]=std::exp(x)+.05*std::sin(31*x);samples[j].value[0]=values[j];}
auto dense=[&](bool gather){for(int k=1;k<m;k+=2){adaptive_detail::Sum s;for(int j=1;j<M;++j)s.add(r.interp[(k/2)*(M-1)+j-1]*(gather?values[j]:samples[j].value[0]));predicted[k]=s.get();}};
dense(true);auto reference=predicted;prediction_detail::sine_predict(m,values.data(),predicted.data());double err=0;for(int k=1;k<m;k+=2)err=std::max(err,std::fabs(reference[k]-predicted[k]));
int reps=2000;auto time=[&](auto f){auto t=std::chrono::steady_clock::now();for(int it=0;it<reps;++it){f();sink+=predicted[1+(it%(m/2))*2];}return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()/reps;};
std::cout<<level<<' '<<m-1<<' '<<time([&]{dense(false);})<<' '<<time([&]{dense(true);})<<' '<<time([&]{prediction_detail::sine_predict(m,values.data(),predicted.data());})<<' '<<err<<'\n';}
return sink==123456?1:0;}
