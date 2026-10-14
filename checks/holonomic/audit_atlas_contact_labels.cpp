#define main atlas_trajectory_main
#include "../../benchmarks/holonomic/bench_radial_atlas_trajectory.cpp"
#undef main
int main(int argc,char**argv){
 if(argc!=2)return 2;std::ifstream in(argv[1]);InputRow row;std::string line;std::cout<<std::setprecision(17);
 while(std::getline(in,line)){if(!read_row(line,row))continue;auto p=LensParams{row.time,row.y,row.rho,1/row.q,row.s,true};auto pf=PrimaryFrame::from(p);
  RadialAtlasWorkspace ws;RadialAtlasConfig cfg;auto atlas=build_radial_event_atlas(pf,cfg,ws);std::vector<Cplx<__float128>> roots;auto old=classify_cells(pf,nullptr,&roots,true);
  std::cout<<"CASE "<<row.case_id<<' '<<row.profile<<' '<<row.d_bin_index<<' '<<row.epoch_index<<" atlas="<<atlas_status_name(atlas.status)<<" contacts="<<atlas.contacts.size()<<'\n';
  for(const auto& c:atlas.contacts){const RadialEvent* best=nullptr;double distance=INFINITY;
   for(const auto& e:old.events)if(e.kind=="physical_real"||e.kind=="physical_complex"){double d=std::fabs(e.radius-double(c.R_center));if(d<distance){distance=d;best=&e;}}
   std::cout<<"CONTACT "<<c.id<<" R="<<double(c.R_center)<<" lo="<<c.R_center.lo<<" chart="<<c.uniqueness.chart<<" s="<<double(c.s_center)<<" closest_old="<<(best?best->kind:"none")<<" delta_R="<<distance;
   if(best&&best->kind=="physical_complex"){
    auto direct=re_detail::probe_double_root(best->radius,pf);auto pc=boundary_quartic(best->radius,pf);QuarticCoeffs reciprocal;
    for(int k=0;k<5;++k)reciprocal.p[k]=(k%2?-1:1)*pc.p[4-k];
    auto candidate=re_detail::quartic_fold_subresultant_seed(reciprocal);
    std::cout<<" direct_probe_real="<<direct.physically_real<<" direct_residual="<<direct.normalized_residual<<" reciprocal_candidate_valid="<<candidate.valid<<" reciprocal_candidate_residual="<<candidate.normalized_residual;
   }
   std::cout<<'\n';
  }
  for(size_t i=0;i<atlas.contacts.size();++i)for(size_t j=0;j<i;++j){const auto& a=atlas.contacts[i];const auto& b=atlas.contacts[j];if(a.R.lo>b.R.hi||b.R.lo>a.R.hi)continue;
   std::cout<<"OVERLAPPING_R "<<a.id<<' '<<b.id<<" original_union_identity="<<atlas_detail::same_contact(pf,a,b)<<'\n';
  }
 }
}
