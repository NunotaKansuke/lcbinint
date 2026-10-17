#pragma once
#include <array>
#include <cmath>
#include <complex>
#include "lcbinint/magnification/algebraic_boundary/nested_fejer2.hpp"
namespace lcbinint::algebraic_boundary::prediction_detail {
// Immutable FFT geometry: depends only on the dyadic node count.
struct Plan {
    int n=0;
    std::array<int,512> reverse{};
    std::array<std::complex<double>,256> root{};
    explicit Plan(int size=2):n(size) {
        int bits=0;while((1<<bits)<n)++bits;
        for(int j=0;j<n;++j) {int r=0,x=j;for(int b=0;b<bits;++b){r=(r<<1)|(x&1);x>>=1;}reverse[j]=r;}
        for(int j=0;j<n/2;++j){const double a=-2*3.14159265358979323846*j/n;root[j]={std::cos(a),std::sin(a)};}
    }
};
inline const Plan& plan(int n) {
    static const std::array<Plan,9> plans={Plan(2),Plan(4),Plan(8),Plan(16),Plan(32),Plan(64),Plan(128),Plan(256),Plan(512)};
    int i=0;while((2<<i)<n)++i;return plans[i];
}
inline void fft(std::array<std::complex<double>,512>& a,int n) {
    const auto& p=plan(n);
    for(int j=0;j<n;++j)if(j<p.reverse[j])std::swap(a[j],a[p.reverse[j]]);
    for(int size=2;size<=n;size*=2)for(int base=0;base<n;base+=size)
        for(int j=0;j<size/2;++j){const auto v=a[base+j+size/2]*p.root[j*n/size],u=a[base+j];a[base+j]=u+v;a[base+j+size/2]=u-v;}
}
// Same polynomial as barycentric interpolation, in exact arithmetic.
// fine_m=2*coarse_m. Coarse values indexed1..coarse_m-1, output indexed
//1..fine_m-1 (only odd entries are needed by the detail estimator).
inline void sine_predict(int fine_m,const double* values,double* predicted) {
    const int M=fine_m/2;
    std::array<std::complex<double>,512> work{};
    const auto& coarse_plan=plan(2*M);
    for(int j=1;j<M;++j){const double y=values[j]*(-coarse_plan.root[j].imag());work[j]=y;work[2*M-j]=-y;}
    fft(work,2*M);
    std::array<double,128> coeff{};
    for(int j=1;j<M;++j)coeff[j]=-work[j].imag()/M;
    work.fill({0,0});
    for(int j=1;j<M;++j){work[j]=coeff[j];work[4*M-j]=-coeff[j];}
    fft(work,4*M);
    const auto& fine_plan=plan(4*M);
    for(int j=1;j<fine_m;j+=2)predicted[j]=-.5*work[j].imag()/(-fine_plan.root[j].imag());
}
}
