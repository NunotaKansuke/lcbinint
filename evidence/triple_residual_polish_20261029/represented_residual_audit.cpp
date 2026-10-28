#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/model/triple_lens_geometry.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
using namespace lcbinint;using namespace lcbinint::magnification;
int main(){int id,n;double s,q,sep,ang,q2,t0,u0,tE,alpha;std::cout<<std::setprecision(17);
 while(std::cin>>id>>s>>q>>sep>>ang>>q2>>t0>>u0>>tE>>alpha>>n){
  const auto g=model::make_triple_lens_geometry_vbm(s,q,sep,ang,q2);PointSourceMagnifier solver;
  for(int j=0;j<n;++j){double t;std::cin>>t;const double tau=(t-t0)/tE;SourcePosition source{tau*std::cos(alpha)-u0*std::sin(alpha),tau*std::sin(alpha)+u0*std::cos(alpha)};const auto& images=solver.triple_image_candidates_cached(g,source);int count=0,bad=0,local=0,positive=0;double maximum=0,recomputed=0;int recomputed_bad=0;
   for(const auto&i:images)if(i.physical){++count;positive+=i.jacobian_determinant>0;local+=i.local_origin>=0;if(!std::isfinite(i.residual)||i.residual>1e-7)++bad;else maximum=std::max(maximum,i.residual);
    using C=std::complex<long double>;C f;
    if(i.local_origin>=0){const auto o=g.lens_positions[i.local_origin];f=C((long double)o.x-source.x,(long double)o.y-source.y)+C(i.local_offset.x,i.local_offset.y);}
    else f=C((long double)i.position.x-source.x,(long double)i.position.y-source.y);
    for(int k=0;k<3;++k){const auto d=triple_image_separation(g,i,k);f-=(long double)g.masses[k]/std::conj(C(d.real(),d.imag()));}
    const double error=double(std::abs(f));if(!std::isfinite(error)||error>1e-7)++recomputed_bad;else recomputed=std::max(recomputed,error);
   }
   std::cout<<"{\"case\":"<<id<<",\"epoch\":"<<j<<",\"count\":"<<count<<",\"bad_physical_residuals\":"<<bad<<",\"local\":"<<local<<",\"positive\":"<<positive<<",\"maximum_valid_residual\":"<<maximum<<",\"recomputed_bad\":"<<recomputed_bad<<",\"maximum_recomputed_residual\":"<<recomputed<<"}\n";
  }
 }
}
