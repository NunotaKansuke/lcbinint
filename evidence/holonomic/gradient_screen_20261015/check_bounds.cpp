#include <fused_geometry_screen.hpp>
#include <random>
#include <iostream>
using namespace lcbinint::holonomic_fused;
using namespace geometry_experiment;
int main(){std::mt19937_64 rng(764332);int checks=0;for(int k=0;k<100000;++k){auto bits=rng();double x;std::memcpy(&x,&bits,8);if(std::isnan(x))continue;double a=screen_up(x),b=std::nextafter(x,INFINITY),c=screen_down(x),d=std::nextafter(x,-INFINITY);if(std::memcmp(&a,&b,8)||std::memcmp(&c,&d,8)){std::cerr<<"rounding mismatch\n";return 1;}++checks;}
for(double x:{0.,-0.,double(INFINITY),double(-INFINITY),std::numeric_limits<double>::denorm_min(),-std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::max(),-std::numeric_limits<double>::max()}){double a=screen_up(x),b=std::nextafter(x,INFINITY),c=screen_down(x),d=std::nextafter(x,-INFINITY);if(std::memcmp(&a,&b,8)||std::memcmp(&c,&d,8))return 2;++checks;}
for(int k=0;k<1000;++k){LensParams p{(double(rng()%10000)/10000-.5),(double(rng()%10000)/10000-.5),std::pow(10.,-1-int(rng()%8)),std::pow(10.,int(rng()%17)-8),.1+double(rng()%10000)/1000,true};auto pf=PrimaryFrame::from(p);double center=double(rng()%10000)/1000,radius=.001+double(rng()%1000)/10000;auto a=ScreenBlocks<ScreenInterval>(pf).disk(pf,center,radius);auto b=ScreenBlocks<positive_detail::Interval<double>>(pf).disk(pf,center,radius);auto estimated=ScreenBlocks<ScreenEstimate>(pf).disk(pf,center,radius);auto reference=ScreenBlocks<positive_detail::Interval<__float128>>(pf).disk(pf,center,radius);
for(int j=0;j<=14;++j){auto e=estimated.c[j];if((__float128)e.c-e.error()>reference.c[j].lo||(__float128)e.c+e.error()<reference.c[j].hi){std::cerr<<"estimate enclosure failure "<<k<<" coefficient "<<j<<"\n";return 5;}}
for(int j=0;j<=14;++j)if(a.c[j].lo!=b.c[j].lo||a.c[j].hi!=b.c[j].hi){std::cerr<<"coefficient mismatch\n";return 3;}}
std::cout<<checks<<" IEEE steps and 15000 interval coefficients agree\n";}
