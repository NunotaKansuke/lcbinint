#include <iostream>
#ifdef AXIS
#include "radial_events_axis.hpp"
#else
#include "lcbinint/magnification/algebraic_boundary/triple/radial_events.hpp"
#endif
#include "lcbinint/magnification/algebraic_boundary/triple/frame.hpp"
using namespace lcbinint::algebraic_boundary::triple;
int main(){int id,j;double s,q,sep,ang,q2,X,Y,rho;while(std::cin>>id>>j>>s>>q>>sep>>ang>>q2>>X>>Y>>rho){double off=q*s/(1+q),den=1+q+q2;std::array<Complex,3> l{Complex(off,0),Complex(-s/(1+q),0),Complex(off,0)-sep*std::polar(1.,ang)};std::array<double,3> m{1/den,q/den,q2/den};const std::array<std::array<int,3>,3> orders{{{0,1,2},{1,0,2},{2,0,1}}};for(int k=0;k<3;++k){auto o=orders[k];std::array<Complex,3>a{l[o[0]],l[o[1]],l[o[2]]};auto f=primary_frame(a,{X,Y},rho,m[o[1]],m[o[2]]);
#ifdef AXIS
 event_detail::precision_axis_fallbacks=0;
#endif
 auto p=radial_event_plan(f.geometry,65536,0);std::cout<<"{\"case\":"<<id<<",\"epoch\":"<<j<<",\"frame\":"<<k<<",\"completed\":"<<p.completed<<",\"reason\":\""<<p.reason<<"\",\"boxes\":"<<p.boxes;
#ifdef AXIS
 std::cout<<",\"axis_fallbacks\":"<<event_detail::precision_axis_fallbacks;
#endif
 std::cout<<"}\n"<<std::flush;}}}
