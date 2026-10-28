#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/model/triple_lens_geometry.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
using namespace lcbinint;using namespace lcbinint::magnification;using C=std::complex<long double>;
int main(){int id,n;double s,q,sep,ang,q2,t0,u0,tE,alpha;std::cout<<std::setprecision(21);
while(std::cin>>id>>s>>q>>sep>>ang>>q2>>t0>>u0>>tE>>alpha>>n){auto g=model::make_triple_lens_geometry_vbm(s,q,sep,ang,q2);
for(int j=0;j<n;++j){double t;std::cin>>t;if((id==147&&j!=3)||(id==482&&j!=1))continue;double tau=(t-t0)/tE;C y(tau*cos(alpha)-u0*sin(alpha),tau*sin(alpha)+u0*cos(alpha));std::vector<C> roots;
auto eval=[&](C z){C f=z-y,k{};for(int a=0;a<3;++a){C d=z-C(g.lens_positions[a].x,g.lens_positions[a].y);f-=(long double)g.masses[a]/conj(d);k+=(long double)g.masses[a]/(conj(d)*conj(d));}return std::make_pair(f,k);};
auto attempt=[&](C z){for(int it=0;it<100;++it){auto[f,k]=eval(z);long double det=1-norm(k);if(!std::isfinite(abs(f))||!std::isfinite(det)||det==0)return;C dz=(f-k*conj(f))/det;z-=dz;if(abs(dz)<1e-19L*(1+abs(z)))break;}auto[f,k]=eval(z);if(!(abs(f)<1e-12L))return;for(C old:roots)if(abs(old-z)<1e-8L)return;roots.push_back(z);};
for(int lens=0;lens<3;++lens)for(int r=-10;r<=3;++r)for(int a=0;a<16;++a){long double angle=2*acosl(-1.L)*a/16;long double radius=powl(10.L,r*.5L)*sqrtl(g.masses[lens]);attempt(C(g.lens_positions[lens].x,g.lens_positions[lens].y)+radius*C(cosl(angle),sinl(angle)));}
int positive=0;std::cout<<"case "<<id<<" epoch "<<j<<" roots "<<roots.size()<<"\n";for(C z:roots){auto[f,k]=eval(z);positive+=1-norm(k)>0;std::cout<<"root "<<z.real()<<" "<<z.imag()<<" residual "<<abs(f)<<" det "<<1-norm(k)<<"\n";}std::cout<<"positive "<<positive<<"\n";
PointSourceMagnifier sol;for(auto i:sol.triple_image_candidates_cached(g,{double(y.real()),double(y.imag())}))if(i.physical)std::cout<<"native "<<i.position.x<<" "<<i.position.y<<" residual "<<i.residual<<" det "<<i.jacobian_determinant<<"\n";
}}}
