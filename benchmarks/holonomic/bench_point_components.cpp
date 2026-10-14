#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/binary_caustic_enclosure.hpp"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
using Clock=std::chrono::steady_clock;
using namespace lcbinint::magnification;
volatile double sink=0.;
template<class F> double measure(F fn) {
 std::vector<double> samples;
 for(int j=0;j<101;++j){auto start=Clock::now(); for(int k=0;k<10;++k)fn(); samples.push_back(std::chrono::duration<double,std::micro>(Clock::now()-start).count()/10.);}
 std::sort(samples.begin(),samples.end()); return samples[50];
}
int main(){
 std::cout<<"cached safety 201 epochs us "<<measure([]{PointSourceMagnifier p;for(int i=0;i<201;++i)sink=p.binary_safety_diagnostic_cached(1.2,100.,{-1.+.01*i,.3}).magnification;})<<'\n';
 std::cout<<"cached scalar roots 201 epochs us "<<measure([]{PointSourceMagnifier p;for(int i=0;i<201;++i)sink=p.binary_mag0_cached(1.2,100.,{-1.+.01*i,.3}).magnification;})<<'\n';
 std::cout<<"build enclosure us "<<measure([]{auto cover=detail::build_binary_caustic_enclosure(1.2,100.);sink=cover.size();})<<'\n';
 auto cover=detail::build_binary_caustic_enclosure(1.2,100.);
 std::cout<<"disks "<<cover.size()<<'\n';
 std::cout<<"scan enclosure 201 epochs us "<<measure([&]{for(int i=0;i<201;++i)sink=detail::binary_caustic_enclosure_clearance(cover,{-1.+.01*i,.3});})<<'\n';
}
