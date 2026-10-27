#pragma once
#include "patterson_rule.hpp"
namespace lcbinint::algebraic_boundary::triple {
// Positive nested subset3 -> G7 -> K15, degrees3/13/23.
// Every earlier value/geometry sample belongs to the next rule.
// K15 constants match Boost.Math gauss_kronrod_detail<double,15,1>.
inline const PattersonRule& gradient_kronrod_rule(int level){

 static const auto fine=[](){PattersonRule q(4);const double x[]={0,.207784955007898468,.405845151377397167,.586087235467691130,.741531185599394440,.864864423359769073,.949107912342758525,.991455371120812639};const double w[]={.209482141084727828,.204432940075298892,.190350578064785410,.169004726639267903,.140653259715525919,.104790010322250184,.0630920926299785533,.0229353220105292250};
  for(int k=0;k<=7;++k){q.x[7-k]=x[k];q.x[7+k]=-x[k];q.w[7-k]=q.w[7+k]=w[k];}q.x[7]=0;q.interp.assign(8*7,0);std::array<long double,7> bary{};
  for(int j=0;j<7;++j){bary[j]=1;for(int k=0;k<7;++k)if(k!=j)bary[j]/=(long double)q.x[2*j+1]-q.x[2*k+1];}
  for(int k=0;k<15;k+=2){long double denominator=0;for(int j=0;j<7;++j){const long double value=bary[j]/((long double)q.x[k]-q.x[2*j+1]);q.interp[(k/2)*7+j]=double(value);denominator+=value;}for(int j=0;j<7;++j)q.interp[(k/2)*7+j]/=double(denominator);}return q;}();
 if(level==1){static const auto initial=[&](){PattersonRule q(2);const double a=fine.x[3],weight=1/(3*a*a);q.x={a,0,-a};q.w={weight,2-2*weight,weight};q.interp.clear();return q;}();return initial;}
 if(level==3)return fine;
 if(level==2){static const auto coarse=[&](){PattersonRule q(3);const double w[]={.417959183673469388,.381830050505118945,.279705391489276668,.129484966168869693};for(int j=0;j<7;++j){q.x[j]=fine.x[2*j+1];q.w[j]=w[std::abs(j-3)];}q.interp.clear();return q;}();return coarse;}
 throw std::invalid_argument("gradient Kronrod level");
}
}
