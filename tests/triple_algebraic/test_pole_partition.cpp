#include "lcbinint/magnification/algebraic_boundary/triple/pole_partition.hpp"
#include <iostream>
#include <stdexcept>
using namespace lcbinint::algebraic_boundary::triple;
int main() {
    const Geometry p{.01,.001,.003,1e-4,1e-6,100.,200.,1e-6};
    using L=std::complex<long double>;
    const auto lenses=p.lenses();
    const std::array<long double,3> mass{1.L-p.m1-p.m2,p.m1,p.m2};
    for(int pole:{1,2}) {
        const auto proof=prove_pole_partition(p,pole,.1);
        if(!proof)throw std::runtime_error("wide pole partition refused");
        if(!(proof->radius_lo>0 && proof->radius_hi<proof->region.radius))
            throw std::runtime_error("partition domain ordering");
        for(long double fraction:{0.L,.5L,1.L})for(int k=0;k<64;++k) {
            const auto source=L(p.X,p.Y)+std::polar(fraction*p.rho,2*acosl(-1.L)*k/64);
            const L origin(lenses[pole].real(),lenses[pole].imag());L delta{};
            for(int iteration=0;iteration<30;++iteration) {
                L H=origin-source+delta;
                for(int j=0;j<3;++j)if(j!=pole) {
                    const L lens(lenses[j].real(),lenses[j].imag());
                    H-=mass[j]/std::conj(origin-lens+delta);
                }
                delta=mass[pole]/std::conj(H);
            }
            const auto radius=std::abs(delta);
            if(!(radius>=proof->radius_lo && radius<=proof->radius_hi))
                throw std::runtime_error("original-equation root outside radial enclosure");
            L field=origin-source+delta,shear{};
            for(int j=0;j<3;++j) {
                const L lens(lenses[j].real(),lenses[j].imag());
                const auto inverse=1.L/std::conj(origin-lens+delta);
                field-=mass[j]*inverse;shear+=mass[j]*inverse*inverse;
            }
            const auto determinant=1-std::norm(shear);
            if(!(std::abs(field)<1e-12L && determinant<0 &&
                 1/std::abs(determinant)<=proof->magnification_upper))
                throw std::runtime_error("pole branch residual or flux bound");
        }
    }
    if(prove_pole_partition(p,1,100.) || prove_pole_partition(p,1,0.) ||
       prove_pole_partition(p,-1,.1))throw std::runtime_error("invalid partition accepted");
    auto contact=p;contact.X=99.990001;contact.Y=0;
    if(prove_pole_partition(contact,1,.1))throw std::runtime_error("unproved pole contact accepted");
    auto bad=p;bad.m1=std::nextafter(1.,0.);bad.m2=2e-16;
    if(prove_pole_partition(bad,1,.1))throw std::runtime_error("nonpositive primary accepted");
    std::cout<<"PASS: pole contraction, annular complement and original-equation bounds\n";
}
