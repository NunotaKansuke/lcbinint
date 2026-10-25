from pathlib import Path
s=Path('evidence/triple_resultant_sparse/support-lens-basis.hpp').read_text()
start=s.index('inline std::array<Ball,29> support_raw(')
end=s.index(' // Subset determinant:',start)
new='''// Polarize the quadratic Bezout construction in the four source scalars
// (1, X, Y, rho^2-X^2-Y^2). All ten matrix coefficients depend only on lenses.
struct MatrixBasis {
 std::array<double,5> identity;
 using Entry=std::array<Ball,9>;
 std::array<std::array<std::array<Entry,5>,5>,10> coefficients{};
 explicit MatrixBasis(const Geometry&p):identity{p.m1,p.m2,p.a,p.bx,p.by} {
  const auto&base=lens_basis(p);
  const std::array<const std::array<Poly,7>*,4> q{{&base.constant,&base.x,&base.y,&base.disk}};
  std::array<std::array<Poly,6>,4>u{},v{};
  for(int h=0;h<4;++h)for(int i=0;i<6;++i){u[h][i]=Poly(i+1)*(*q[h])[i+1];v[h][i]=Poly(6-i)*(*q[h])[i];}
  constexpr int valuation[5]={2,1,0,0,0},weight[5]={2,3,4,3,2};
  int phase[5][5];for(auto&row:phase)for(auto&x:row)x=-1;
  phase[0][0]=phase[4][0]=phase[1][1]=phase[3][1]=phase[2][2]=0;
  phase[0][4]=phase[1][3]=1;phase[4][4]=phase[3][3]=3;
  int term=0;
  for(int a=0;a<4;++a)for(int b=a;b<4;++b,++term){
   std::array<std::array<Poly,5>,5>bezout{};
   for(int i=1;i<=5;++i)for(int j=0;j<i;++j){
    Poly d=u[a][i]*v[b][j]-u[a][j]*v[b][i];
    if(a!=b)d=d+u[b][i]*v[a][j]-u[b][j]*v[a][i];
    for(int k=0;k<i-j;++k)bezout[i-1-k][j+k]=bezout[i-1-k][j+k]+d;
   }
   for(int i=0;i<5;++i)for(int j=i;j<5;++j){
    auto&raw=coefficients[term][i][j];const int degree=weight[i]+weight[j];
    for(int k=0;k<5;++k)if(phase[k][i]>=0)for(int l=0;l<5;++l)if(phase[l][j]>=0){
     const int shift=valuation[k]+valuation[l],ph=(phase[k][i]+phase[l][j])%4;
     for(int n=0;n<=degree;++n){const C z=bezout[k][l].c[n+shift];Ball value=ph==0?z.re:ph==1?-z.im:ph==2?-z.re:z.im;raw[n]=raw[n]+value;}
    }
   }
  }
 }
};
inline const MatrixBasis& matrix_basis(const Geometry&p) {
 thread_local std::optional<MatrixBasis> cached;
 const std::array<double,5> identity{p.m1,p.m2,p.a,p.bx,p.by};
 if(!cached || cached->identity!=identity)cached.emplace(p);
 return *cached;
}
inline std::array<Ball,29> support_raw(const Geometry&p,double lo,double hi){
 const auto&basis=matrix_basis(p);
 const std::array<Ball,4> source{{Ball(1),Ball(p.X),Ball(p.Y),Ball(p.rho)*Ball(p.rho)-Ball(p.X)*Ball(p.X)-Ball(p.Y)*Ball(p.Y)}};
 std::array<Ball,10> monomials;int term=0;
 for(int a=0;a<4;++a)for(int b=a;b<4;++b)monomials[term++]=source[a]*source[b];
 constexpr int weight[5]={2,3,4,3,2};
 std::array<std::array<RealPoly,5>,5>matrix{};
 for(int i=0;i<5;++i)for(int j=i;j<5;++j){
  const int degree=weight[i]+weight[j];RealPoly raw;raw.degree=degree;
  for(int h=0;h<10;++h)for(int n=0;n<=degree;++n)raw.c[n]=raw.c[n]+monomials[h]*basis.coefficients[h][i][j][n];
  RealPoly current(raw.c[degree]);
  for(int step=1;step<=degree;++step){RealPoly next;next.degree=step;
   for(int k=0;k<=step;++k){if(k<step)next.c[k]=next.c[k]+current.c[k]*Ball(lo);if(k)next.c[k]=next.c[k]+current.c[k-1]*Ball(hi);next.c[k]=next.c[k]+raw.c[degree-step]*Ball(angular_detail::choose(step,k));}current=next;
  }matrix[i][j]=matrix[j][i]=current;
 }
'''
s=s[:start]+new+s[end:]
Path('evidence/triple_resultant_sparse/support-matrix-basis.hpp').write_text(s)
