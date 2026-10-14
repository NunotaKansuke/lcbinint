#pragma once

// Direct Hermite reduction of -t^k Q_x/(2 y^3), y^2=Q.
// Solve Q C_k + Q S'_k - Q' S_k/2 = -t^k Q_x/2 in coefficient space.
// Unlike inverse-Q' modulo Q this never divides by the leading coefficient.
// Constant row/column equilibration is not differentiated. No period samples.
#include "lcbinint/magnification/holonomic/gm_connection.hpp"

namespace lcbinint::holonomic {
template<int Order, class Scalar>
GmConnectionJet<Order, Scalar> gm_direct_connection_jet(
    double R, const GmParams<Scalar>& p, double scale) {
    using Jet=GmConnectionJet<Order,Scalar>;
    using S=GmSeries<Order,Scalar>;
    constexpr int N=15;
    Jet out;
    if (!(scale>0) || !std::isfinite(scale)) return out;
    GmPoly<GmSeries<Order+1,Scalar>,kGmPolyCapacity> qr;
    const auto q=gm_q_series_raw<Order+1,Scalar>(R,p,qr);
    out.q.deg=out.qR.deg=8;
    for(int i=0;i<=8;++i){double power=1;
        for(int n=0;n<=Order;++n){
            out.q.c[i].c[n]=q.c[i].c[n]*Scalar(power);
            out.qR.c[i].c[n]=qr.c[i].c[n]*Scalar(power*scale);
            power*=scale;
        }
    }
    out.degree_ok=gm_abs_value(out.q.c[8].c[0])>0;
    if(!out.degree_ok) return out;
    std::array<std::array<S,N>,N> A{};
    for(int i=0;i<=8;++i){
        for(int j=0;j<7;++j) A[i+j][j]=out.q.c[i];
        for(int j=0;j<8;++j) if(i+j>0)
            A[i+j-1][7+j]=gm_series_scale_scalar(out.q.c[i],Scalar(j-0.5*i));
    }
    std::array<double,N> rows{},cols{};
    for(int i=0;i<N;++i){
        for(int j=0;j<N;++j) rows[i]=std::max(rows[i],gm_abs_value(A[i][j].c[0]));
        if(!(rows[i]>0))return out;
    }
    for(int j=0;j<N;++j){
        for(int i=0;i<N;++i) cols[j]=std::max(cols[j],gm_abs_value(A[i][j].c[0])/rows[i]);
        if(!(cols[j]>0))return out;
    }
    std::array<std::array<Scalar,N>,N> lu{};
    std::array<int,N> piv{};
    for(int i=0;i<N;++i){piv[i]=i;for(int j=0;j<N;++j){
        A[i][j]=gm_series_scale_scalar(A[i][j],Scalar(1/rows[i]/cols[j]));
        lu[i][j]=A[i][j].c[0];
    }}
    out.matrix_pivot_rel=1;
    for(int k=0;k<N;++k){int best=k;
        for(int i=k+1;i<N;++i)if(gm_abs_value(lu[i][k])>gm_abs_value(lu[best][k]))best=i;
        double pivot=gm_abs_value(lu[best][k]);
        if(!(pivot>1e-30)||!std::isfinite(pivot))return out;
        out.matrix_pivot_rel=std::min(out.matrix_pivot_rel,pivot);
        std::swap(lu[k],lu[best]);std::swap(piv[k],piv[best]);
        for(int i=k+1;i<N;++i){lu[i][k]=lu[i][k]/lu[k][k];
            for(int j=k+1;j<N;++j)lu[i][j]=lu[i][j]-lu[i][k]*lu[k][j];}
    }
    out.squarefree=true;
    double residual=0;
    for(int k=0;k<7;++k){
        std::array<S,N> x{},rhs{};
        for(int i=0;i<=8;++i)rhs[i+k]=gm_series_scale_scalar(out.qR.c[i],Scalar(-0.5/rows[i+k]));
        for(int n=0;n<=Order;++n){
            std::array<Scalar,N> b{},sol{};
            for(int i=0;i<N;++i){b[i]=rhs[i].c[n];
                for(int m=1;m<=n;++m)for(int j=0;j<N;++j)
                    b[i]=b[i]-A[i][j].c[m]*x[j].c[n-m];}
            for(int i=0;i<N;++i){sol[i]=b[piv[i]];
                for(int j=0;j<i;++j)sol[i]=sol[i]-lu[i][j]*sol[j];}
            for(int i=N-1;i>=0;--i){
                for(int j=i+1;j<N;++j)sol[i]=sol[i]-lu[i][j]*sol[j];
                sol[i]=sol[i]/lu[i][i];x[i].c[n]=sol[i];}
        }
        // Componentwise backward error of the unreduced identity.
        for(int i=0;i<N;++i)for(int n=0;n<=Order;++n){
            Scalar r=-rhs[i].c[n];double mag=gm_abs_value(rhs[i].c[n]);
            for(int j=0;j<N;++j)for(int m=0;m<=n;++m){
                Scalar term=A[i][j].c[m]*x[j].c[n-m];r=r+term;mag+=gm_abs_value(term);}
            residual=std::max(residual,gm_abs_value(r)/(1+mag));
        }
        for(int j=0;j<7;++j)out.C[k][j]=gm_series_scale_scalar(x[j],Scalar(1/cols[j]));
        out.S[k].deg=7;
        for(int j=0;j<8;++j)out.S[k].c[j]=gm_series_scale_scalar(x[j+7],Scalar(1/cols[j+7]));
    }
    out.identity_residual=residual;
    out.finite=true;
    for(auto& row:out.C)for(auto& s:row)out.finite=out.finite&&gm_series_finite(s);
    out.quality_ok=out.finite && residual<=gm_quality_limit(Scalar(0));
    out.ok=out.quality_ok;
    return out;
}
} // namespace lcbinint::holonomic
