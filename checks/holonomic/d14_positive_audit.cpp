// Independent exact-arithmetic audit emitter; not a timing benchmark.
#define main trajectory_runner_main
#include "../../benchmarks/holonomic/bench_d14_positive_trajectory.cpp"
#undef main
static void qprint(__float128 v){char b[128];quadmath_snprintf(b,sizeof b,"%Qa",v);std::cout<<b;}
template<class I> void coefficients(const PrimaryFrame& pf,int tier){
 auto p=positive_detail::polynomial<I>(pf);
 for(int j=0;j<=p.degree;++j){auto b=positive_detail::bounds(p.c[j]);
  std::cout<<"C "<<tier<<' '<<j<<' ';qprint(b.lo);std::cout<<' ';qprint(b.hi);std::cout<<'\n';}
}
static void emit(const PrimaryFrame& pf,int id){
 const double w=std::hypot(pf.X,pf.Y)+pf.rho;
 const double rm=id==-1 ? 3.0 : 0.5*(pf.a+w+std::hypot(pf.a-w,2.0))+1e-12;
 auto r=positive_d14_roots(pf,rm);
 std::cout<<"P "<<id<<' '<<std::hexfloat<<pf.a<<' '<<pf.m0<<' '<<pf.X<<' '<<pf.Y<<' '<<pf.rho<<std::defaultfloat
  <<' '<<r.domain_exponent<<' '<<int(r.assurance)<<' '<<r.root_count<<' '<<r.stats.reason<<' '<<r.stats.chain_tier<<'\n';
 coefficients<positive_detail::Interval<double>>(pf,0);
 coefficients<positive_detail::Ball>(pf,1);
 coefficients<positive_detail::Interval<__float128>>(pf,2);
 for(unsigned j=0;j<r.root_count;++j){std::cout<<"R ";qprint(r.roots[j].v_lo);std::cout<<' ';qprint(r.roots[j].v_hi);std::cout<<'\n';}
}
int main(int argc,char**argv){
 if(argc!=2)return 2;std::ifstream in(argv[1]);std::string line;InputRow row;int k=0;
 emit(PrimaryFrame::from(LensParams{0.3,0.2,0.01,2,1,true}),-1);
 while(std::getline(in,line)){
  if(!read_row(line,row))continue;if(k++%451)continue;
  emit(PrimaryFrame::from(LensParams{row.time,row.y,row.rho,1.0/row.q,row.s,true}),k);
 }
}
