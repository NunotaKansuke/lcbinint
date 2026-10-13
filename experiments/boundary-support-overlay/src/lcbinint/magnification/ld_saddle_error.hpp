#pragma once
#include <array>
#include <cstdlib>
#include <cmath>
#include <algorithm>
namespace lcbinint::magnification::detail {
inline double saddle_log_basis(double t) {return t==0?0:t*std::log(std::abs(t));}
inline long double saddle_log_primitive(long double t) {return t==0?0:t*t*(std::log(std::abs(t))/2-.25L);}
// Whole-lattice defect for t log|t|. Remove smooth finite-tail midpoint
// Euler--Maclaurin terms; no lens samples or resolution comparison.
inline double saddle_log_moment(double phase) {
 constexpr int count=16;const long double p=phase,a=-count-.5L-p,b=count+.5L-p;
 long double value=saddle_log_primitive(b)-saddle_log_primitive(a);
 for(int k=-count;k<=count;++k) {const long double t=k-p;if(t!=0)value-=t*std::log(std::abs(t));}
 value-=(std::log(std::abs(b))-std::log(std::abs(a)))/24;
 value+=7*(-1/(b*b)+1/(a*a))/5760;
 value-=31*(-6/(b*b*b*b)+6/(a*a*a*a))/967680;
 return static_cast<double>(value);
}
struct SaddleRowModel {
 bool resolved=false;double signed_residual=0,held_defect=0,quadrature_remainder=0;
 std::array<double,8> predicted{};
};
inline SaddleRowModel fit_saddle_rows(const std::array<double,8>& values,double phase) {
 SaddleRowModel result;if(!(phase>0&&phase<1))return result;
 std::array<std::array<double,4>,8> basis;
 for(int k=0;k<8;++k){const double t=k-3-phase;basis[k]={1,t,t*t,saddle_log_basis(t)};}
 // Hold out the two rows closest to the contact. Fit a regular quadratic
 // background plus the analytically derived singular basis on six others.
 const bool legacy_remainder=std::getenv("LCBININT_DIAGNOSTIC_LEGACY_SADDLE_REMAINDER")!=nullptr;
 const int columns=legacy_remainder?5:9;
 double normal[4][9]{};
 if(!legacy_remainder)for(int j=0;j<4;++j)normal[j][5+j]=1;
 for(int k:{0,1,2,5,6,7})for(int i=0;i<4;++i){for(int j=0;j<4;++j)normal[i][j]+=basis[k][i]*basis[k][j];normal[i][4]+=basis[k][i]*values[k];}
 for(int i=0;i<4;++i){int pivot=i;for(int j=i+1;j<4;++j)if(std::abs(normal[j][i])>std::abs(normal[pivot][i]))pivot=j;
  if(!(std::abs(normal[pivot][i])>1e-12))return result;
  for(int j=i;j<columns;++j)std::swap(normal[i][j],normal[pivot][j]);
  const double div=normal[i][i];for(int j=i;j<columns;++j)normal[i][j]/=div;
  for(int k=0;k<4;++k)if(k!=i){const double factor=normal[k][i];for(int j=i;j<columns;++j)normal[k][j]-=factor*normal[i][j];}
 }
 for(int k=0;k<8;++k){for(int j=0;j<4;++j)result.predicted[k]+=basis[k][j]*normal[j][4];result.held_defect=std::max(result.held_defect,std::abs(result.predicted[k]-values[k]));}
 if(legacy_remainder){result.signed_residual=normal[3][4]*saddle_log_moment(phase);result.resolved=std::isfinite(result.signed_residual)&&std::isfinite(result.held_defect);return result;}
 // The junction pass already keeps |Delta²(actual-model)|/24.
 // Add the next midpoint term from the same residual, instead of treating
 // a row-value discrepancy (including constant/linear background) as flux
 // integration error a second time. This remains an asymptotic indicator.
 for(int k=2;k<=5;++k) {
  const auto residual=[&](int j){return values[j]-result.predicted[j];};
  const double fourth=residual(k-2)-4*residual(k-1)+6*residual(k)-4*residual(k+1)+residual(k+2);
  result.quadrature_remainder+=7*std::abs(fourth)/5760;
 }
 const double moment=saddle_log_moment(phase);
 result.signed_residual=normal[3][4]*moment;
 // Include the two withheld rows by rank-one least-squares updates.
 // Transport their coefficient disagreement through the actual tlog
 // quadrature moment, rather than adding row-value error with unit weight.
 // This is local model sensitivity, not a larger-resolution convergence test.
 const double original_amplitude=normal[3][4];
 for(int k:{3,4}) {
  std::array<double,4> direction{};double residual=values[k],denominator=1;
  for(int j=0;j<4;++j){residual-=basis[k][j]*normal[j][4];for(int l=0;l<4;++l)direction[j]+=normal[j][5+l]*basis[k][l];}
  for(int j=0;j<4;++j)denominator+=basis[k][j]*direction[j];
  if(!(denominator>0&&std::isfinite(denominator)))return {};
  for(int j=0;j<4;++j){normal[j][4]+=direction[j]*residual/denominator;for(int l=0;l<4;++l)normal[j][5+l]-=direction[j]*direction[l]/denominator;}
 }
 result.quadrature_remainder+=std::abs((normal[3][4]-original_amplitude)*moment);
 result.resolved=std::isfinite(result.signed_residual)&&std::isfinite(result.held_defect)&&std::isfinite(result.quadrature_remainder);return result;
}
struct PairedSaddleRowModel {
 bool resolved=false;double signed_residual=0,quadrature_remainder=0,regular_fourth=0;
 std::array<double,13> predicted{};
};
// For a quadratic local level set, lambda=(y-a)(y-b), integrating
// sqrt(x*x+lambda) yields lambda log|lambda|. Both contacts therefore
// share one amplitude; fitting unrelated amplitudes is ill-conditioned
// when their retained footprints overlap.
inline double saddle_square_log_moment(double phase) {
 constexpr int count=16;const long double p=phase,a=-count-.5L-p,b=count+.5L-p;
 const auto primitive=[](long double t){return t==0?0:t*t*t*(std::log(std::abs(t))/3-1.L/9);};
 long double value=primitive(b)-primitive(a);
 for(int k=-count;k<=count;++k){const long double t=k-p;if(t!=0)value-=t*t*std::log(std::abs(t));}
 value-=((2*b*std::log(std::abs(b))+b)-(2*a*std::log(std::abs(a))+a))/24;
 value+=7*(2/b-2/a)/5760;
 value-=31*(4/(b*b*b)-4/(a*a*a))/967680;
 return static_cast<double>(value);
}

// Reuse each logarithm for all three powers needed by a quadratic level set
// with a linearly varying brightness coefficient.
inline std::array<double,3> saddle_log_moments(double phase) {
 constexpr int count=16;const long double a=-count-.5L-phase,b=count+.5L-phase,la=std::log(std::abs(a)),lb=std::log(std::abs(b));
 std::array<long double,3> v={b*b*(lb/2-.25L)-a*a*(la/2-.25L),b*b*b*(lb/3-1.L/9)-a*a*a*(la/3-1.L/9),b*b*b*b*(lb/4-1.L/16)-a*a*a*a*(la/4-1.L/16)};
 for(int k=-count;k<=count;++k){const long double t=k-phase;if(t!=0){long double term=t*std::log(std::abs(t));for(int j=0;j<3;++j){v[j]-=term;term*=t;}}}
 v[0]-=(lb-la)/24;v[0]+=7*(-1/(b*b)+1/(a*a))/5760;v[0]-=31*(-6/(b*b*b*b)+6/(a*a*a*a))/967680;
 v[1]-=(2*b*lb+b-2*a*la-a)/24;v[1]+=7*(2/b-2/a)/5760;v[1]-=31*(4/(b*b*b)-4/(a*a*a))/967680;
 v[2]-=(3*b*b*lb+b*b-3*a*a*la-a*a)/24;v[2]+=7*(6*lb-6*la)/5760;v[2]-=31*(-6/(b*b)+6/(a*a))/967680;
 return {static_cast<double>(v[0]),static_cast<double>(v[1]),static_cast<double>(v[2])};
}
inline PairedSaddleRowModel fit_paired_saddle_rows(const std::array<double,13>& values,int count,double left,double right) {
 PairedSaddleRowModel result;
 if(count<9||count>13||!(left>1&&right<count-2&&left<right))return result;
 std::array<std::array<double,7>,13> basis{};std::array<bool,13> held{};
 held[0]=held[count-1]=true;
 double normal[7][15]{};for(int j=0;j<7;++j)normal[j][8+j]=1;
 int train=0;const double centre=(left+right)/2,d=right-left;
 for(int k=0;k<count;++k) {
  const double t=(k-centre)/d,lambda=(k-left)*(k-right)/(d*d),singular=saddle_log_basis(lambda);
  basis[k]={1,t,t*t,t*t*t,t*t*t*t,singular,t*singular};
  if(held[k])continue;++train;
  for(int i=0;i<7;++i){for(int j=0;j<7;++j)normal[i][j]+=basis[k][i]*basis[k][j];normal[i][7]+=basis[k][i]*values[k];}
 }
 if(train<7)return result;
 for(int i=0;i<7;++i) {
  int pivot=i;for(int j=i+1;j<7;++j)if(std::abs(normal[j][i])>std::abs(normal[pivot][i]))pivot=j;
  if(!(std::abs(normal[pivot][i])>1e-12))return result;
  for(int j=i;j<15;++j)std::swap(normal[i][j],normal[pivot][j]);
  const double div=normal[i][i];for(int j=i;j<15;++j)normal[i][j]/=div;
  for(int k=0;k<7;++k)if(k!=i){const double factor=normal[k][i];for(int j=i;j<15;++j)normal[k][j]-=factor*normal[i][j];}
 }
 for(int k=0;k<count;++k)for(int j=0;j<7;++j)result.predicted[k]+=basis[k][j]*normal[j][7];
 const auto lm=saddle_log_moments(left-std::floor(left)),rm=saddle_log_moments(right-std::floor(right));
 const double moment=(lm[1]+rm[1]+d*(rm[0]-lm[0]))/(d*d);
 const double asymmetric=(lm[2]+rm[2]+1.5*d*(rm[1]-lm[1])+.5*d*d*(lm[0]+rm[0]))/(d*d*d);
 result.signed_residual=normal[5][7]*moment+normal[6][7]*asymmetric;
 result.regular_fourth=24*normal[4][7]/(d*d*d*d);
 for(int k=0;k<count;++k)if(held[k]) {
  std::array<double,7> direction{};double residual=values[k],denominator=1;
  for(int j=0;j<7;++j){residual-=basis[k][j]*normal[j][7];for(int l=0;l<7;++l)direction[j]+=normal[j][8+l]*basis[k][l];}
  for(int j=0;j<7;++j)denominator+=basis[k][j]*direction[j];
  if(!(denominator>0&&std::isfinite(denominator)))return {};
  for(int j=0;j<7;++j){normal[j][7]+=direction[j]*residual/denominator;for(int l=0;l<7;++l)normal[j][8+l]-=direction[j]*direction[l]/denominator;}
 }
 const double updated_signed=normal[5][7]*moment+normal[6][7]*asymmetric;
 const double change=updated_signed-result.signed_residual;
 if(std::getenv("LCBININT_DIAGNOSTIC_LEGACY_PAIRED_CENTRE")) {
 for(int k=2;k+2<count;++k) {
  const auto r=[&](int j){return values[j]-result.predicted[j];};
  result.quadrature_remainder+=7*std::abs(r(k-2)-4*r(k-1)+6*r(k)-4*r(k+1)+r(k+2))/5760;
 }
  result.quadrature_remainder+=std::abs(change);
 } else {
  // The retained-only fit and its held-row enrichment define two local
  // models. Centre their flux predictions and retain half their spread,
  // as for outer endpoint models. Re-evaluate residuals about that centre.
  result.signed_residual+=.5*change;
  result.regular_fourth=.5*(result.regular_fourth+24*normal[4][7]/(d*d*d*d));
  for(int k=0;k<count;++k) {
   double updated=0;for(int j=0;j<7;++j)updated+=basis[k][j]*normal[j][7];
   result.predicted[k]=.5*(result.predicted[k]+updated);
  }
  result.quadrature_remainder=.5*std::abs(change);
  for(int k=2;k+2<count;++k) {
   const auto r=[&](int j){return values[j]-result.predicted[j];};
   result.quadrature_remainder+=7*std::abs(r(k-2)-4*r(k-1)+6*r(k)-4*r(k+1)+r(k+2))/5760;
  }
 }
 result.resolved=std::isfinite(result.signed_residual)&&std::isfinite(result.quadrature_remainder)&&std::isfinite(result.regular_fourth);
 return result;
}
}
