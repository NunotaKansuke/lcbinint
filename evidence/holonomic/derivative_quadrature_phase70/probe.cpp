#include "../../../tests/holonomic_cpp/adaptive_reference.hpp"
#include <iomanip>
#include <iostream>
using C=std::chrono::steady_clock;
struct V {double f=0,d=0,base_us=0,derivative_us=0;bool ok=true;};
V eval(double R,const PrimaryFrame&pf,bool derivative){
 V o;auto start=C::now();auto arcs=reference_arcs(R,pf);
 if(arcs.kind==ArcKind::kEmpty)return o;
 if(arcs.kind==ArcKind::kFull){o.f=kTwoPi*R;o.d=kTwoPi;return o;}
 if(arcs.kind!=ArcKind::kArcs){o.ok=false;return o;}
 std::vector<std::pair<double,double>> endpoints;
 double width=0;
 for(auto a:arcs.arcs){auto l=polish_endpoint(R,a[0],pf),r=polish_endpoint(R,a[1],pf);
 if(!l.reliable||!r.reliable)o.ok=false;
 double w=r.theta-l.theta;if(w<=0)w+=kTwoPi;width+=w;endpoints.push_back({l.theta,r.theta});}
 o.f=R*width;auto mid=C::now();o.base_us=std::chrono::duration<double,std::micro>(mid-start).count();
 if(derivative){double dw=0;for(auto a:endpoints){double t0=std::tan(a.first/2),t1=std::tan(a.second/2);auto dl=endpoint_dR(t0,R,pf),dr=endpoint_dR(t1,R,pf);if(!dl.ok||!dr.ok)o.ok=false;dw+=2*dr.dt_dR/(1+t1*t1)-2*dl.dt_dR/(1+t0*t0);}o.d=width+R*dw;}
 o.derivative_us=std::chrono::duration<double,std::micro>(C::now()-mid).count();return o;
}
int main(int argc,char**argv){
 std::ifstream in(argv[1]);std::string line;std::cout<<std::setprecision(17);
 std::cout<<"name panel kind x g dg fd base_us derivative_us ref64 ref128 q15\n";
 while(std::getline(in,line)){LensParams p;double u,z;int b;std::string name;std::istringstream s(line);if(!(s>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>b>>u>>z>>name)||u!=0)continue;p.barycentric=b;
 auto pf=PrimaryFrame::from(p);auto topo=classify_cells(pf,nullptr,nullptr,true);std::vector<CellPlan>cells;if(!adaptive_detail::restore_physical_cuts(topo,pf,cells))continue;
 auto physical=[&](double R){for(auto&e:topo.events)if(e.radius==R&&e.physically_real&&e.kind=="physical_real")return true;return false;};
 int id=0;for(auto&cell:cells){++id;if(cell.kind==ArcKind::kEmpty||cell.kind==ArcKind::kDegenerate)continue;FoldRadialMap map{cell.r_lo,cell.r_hi,physical(cell.r_lo),physical(cell.r_hi)};
 auto at=[&](double x,bool d){auto a=map(x);auto v=eval(a[0],pf,d);double w=map.b-map.a,t=(1+x)/2,jp=0;
 if(map.left&&map.right)jp=kPi*kPi*w/8*std::cos(kPi*t);else if(map.left)jp=w/2;else if(map.right)jp=-w/2;
 double norm=kPi*p.rho*p.rho;v.d=(v.d*a[1]*a[1]+v.f*jp)/norm;v.f=v.f*a[1]/norm;if(!v.ok)v.f=v.d=NAN;return v;};
 auto integrate=[&](int n){GL gl(n);double q=0;for(int i=0;i<n;++i)q+=gl.w[i]*at(gl.x[i],false).f;return q;};
 double lo=integrate(64),hi=integrate(128),q15=0;auto&r15=fejer_rule(4);for(size_t i=0;i<r15.x.size();++i)q15+=r15.w[i]*at(r15.x[i],false).f;
 for(double x:fejer_rule(3).x){auto v=at(x,true);double h=1e-5;double fd=(at(x+h,false).f-at(x-h,false).f)/(2*h);
 std::cout<<name<<" "<<id<<" "<<(map.left+2*map.right)<<" "<<x<<" "<<v.f<<" "<<v.d<<" "<<fd<<" "<<v.base_us<<" "<<v.derivative_us<<" "<<lo<<" "<<hi<<" "<<q15<<"\n";}
 }
 }
}
