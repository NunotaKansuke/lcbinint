#pragma once
#include "root_pair.hpp"
namespace lcbinint::algebraic_boundary::triple::complex_product_detail {
// Product integration for a complex-conjugate radial fold. The basis keeps
// its known square root exact instead of approximating it by polynomials.
using J=root_pair_detail::Jet<8>;
struct Rule {std::vector<J> weight;std::vector<std::vector<double>> interpolation;};
struct LU {
 int n;std::array<std::array<long double,7>,7> a{};std::array<int,7> pivot{};bool valid=true;
 LU(int count,const std::array<std::array<long double,7>,7>& input):n(count),a(input){
  for(int k=0;k<n;++k){int p=k;for(int j=k+1;j<n;++j)if(std::abs(a[j][k])>std::abs(a[p][k]))p=j;pivot[k]=p;
   if(!(std::abs(a[p][k])>1e-12L)){valid=false;return;}std::swap(a[p],a[k]);
   for(int i=k+1;i<n;++i){a[i][k]/=a[k][k];for(int j=k+1;j<n;++j)a[i][j]-=a[i][k]*a[k][j];}}
 }
 std::array<long double,7> solve(std::array<long double,7> b)const{
  for(int k=0;k<n;++k)std::swap(b[k],b[pivot[k]]);for(int k=0;k<n;++k)for(int i=k+1;i<n;++i)b[i]-=a[i][k]*b[k];
  for(int i=n-1;i>=0;--i){for(int j=i+1;j<n;++j)b[i]-=a[i][j]*b[j];b[i]/=a[i][i];}return b;
 }
};
inline std::array<long double,7> basis(long double x,long double c,long double s,int n,int signature=1){
 std::array<long double,7> b{};const int poly=(n+1)/2;long double power=1;for(int k=0;k<poly;++k){b[k]=power;power*=x;}
 power=std::sqrt((x-c)*(x-c)+signature*s*s);for(int k=poly;k<n;++k){b[k]=power;power*=x;}return b;
}
inline std::optional<Rule> make(const std::vector<J>& nodes,const J& lo,const J& hi,const J& center,const J& scale,int signature=1){
 const int n=int(nodes.size());if(n!=3&&n!=7)return std::nullopt;const int poly=(n+1)/2;
 const J mid=(lo+hi)*J(.5),half=(hi-lo)*J(.5),inverse=root_pair_detail::inverse(half),c=(center-mid)*inverse,s=scale*inverse;
 if(!(half.value>0&&s.value>1e-10&&s.value<1e6))return std::nullopt;
 if(signature<0&&(!(std::abs(c.value)>1)||std::abs(c.value)-1<s.value*(1-1e-13)))return std::nullopt;
 std::vector<J> x;for(const auto& node:nodes)x.push_back((node-mid)*inverse);
 std::array<std::array<long double,7>,7> B{};for(int j=0;j<n;++j){const auto b=basis(x[j].value,c.value,s.value,n,signature);for(int i=0;i<n;++i)B[i][j]=b[i];}LU lu(n,B);if(!lu.valid)return std::nullopt;
 const long double C=c.value,S=s.value,ss=signature*S*S;
 const auto primitives=[&](long double y){const long double root=std::sqrt(std::max(0.L,y*y+ss)),as=signature>0?std::asinh(y/S):std::copysign(std::acosh(std::max(1.L,std::abs(y)/S)),y);return std::array<long double,7>{.5L*(y*root+ss*as),root*root*root/3,(y*root*(2*y*y+ss)-ss*ss*as)/8,as,root,.5L*(y*root-ss*as),root*root*root/3-ss*root};};
 const auto a=primitives(-1-C),b=primitives(1-C);std::array<long double,7> d;for(int k=0;k<7;++k)d[k]=b[k]-a[k];
 std::array<long double,7> M{},Mc{},Ms{};for(int k=0;k<poly;++k)M[k]=k%2?0:2.L/(k+1);
 M[poly]=d[0];Mc[poly]=-d[4];Ms[poly]=signature*S*d[3];
 if(n==7){M[poly+1]=d[1]+C*d[0];Mc[poly+1]=-d[5]-C*d[4];Ms[poly+1]=signature*S*(d[4]+C*d[3]);M[poly+2]=d[2]+2*C*d[1]+C*C*d[0];Mc[poly+2]=-d[6]-2*C*d[5]-C*C*d[4];Ms[poly+2]=signature*S*(d[5]+2*C*d[4]+C*C*d[3]);}
 const auto w=lu.solve(M);long double absolute=0,sum=0;for(int k=0;k<n;++k){absolute+=std::abs(w[k]);sum+=w[k];}if(absolute>8||std::abs(sum-2)>1e-9)return std::nullopt;
 Rule result;result.weight.resize(n);for(int k=0;k<n;++k)result.weight[k].value=double(half.value*w[k]);
 for(int j=0;j<8;++j){std::array<long double,7> rhs{};for(int i=0;i<n;++i)rhs[i]=Mc[i]*c.dp[j]+Ms[i]*s.dp[j];
  for(int k=0;k<n;++k){const long double X=x[k].value,dx=x[k].dp[j],r=std::sqrt((X-C)*(X-C)+ss),dr=((X-C)*(dx-c.dp[j])+signature*S*s.dp[j])/r;long double power=1,previous=0;
   for(int i=0;i<poly;++i){rhs[i]-=w[k]*i*previous*dx;previous=power;power*=X;}
   power=1;previous=0;for(int i=poly;i<n;++i){const int degree=i-poly;rhs[i]-=w[k]*(power*dr+r*degree*previous*dx);previous=power;power*=X;}}
  const auto dw=lu.solve(rhs);for(int k=0;k<n;++k){result.weight[k].dp[j]=double(half.dp[j]*w[k]+half.value*dw[k]);if(!std::isfinite(result.weight[k].dp[j]))return std::nullopt;}
 }
 if(n==7){std::array<std::array<long double,7>,7> coarse{};for(int j=0;j<3;++j){const auto q=basis(x[2*j+1].value,C,S,3,signature);for(int i=0;i<3;++i)coarse[i][j]=q[i];}LU interpolation(3,coarse);if(!interpolation.valid)return std::nullopt;result.interpolation.resize(n);
  for(int k=0;k<n;++k){const auto coefficients=interpolation.solve(basis(x[k].value,C,S,3,signature));result.interpolation[k].resize(3);for(int j=0;j<3;++j)result.interpolation[k][j]=double(coefficients[j]);}
 }return result;
}
}
