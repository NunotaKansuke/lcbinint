#include "lcbinint/magnification/binary_caustic_tree.hpp"
#include <chrono>
#include <iostream>
int main() {
    using namespace lcbinint;
    using namespace magnification::detail;
    for(double q : {.01, 1., 10.}) {
        double a=q<1?-1.2:1.2, m1=1/(1+std::min(q,1/q)), m2=1-m1;
        math::PolynomialRootSolver solver;
        std::vector<SourcePosition> sources;
        for(int k=0;k<24;++k) {
            Complex phase=std::polar(1.,6.283185307179586*(k+.37)/24);
            for(auto z:solver.solve({m2*a*a,-2*m2*a,1.-phase*a*a,2.*phase*a,-phase}).roots) {
                Complex s=z-m1/(std::conj(z)-a)-m2/std::conj(z)-a*m1;
                sources.push_back({s.real(),s.imag()});
            }
        }
        BinaryCausticTree tree;
        int cleared=0;auto start=std::chrono::steady_clock::now();
        for(int repeat=0;repeat<4;++repeat)for(auto s:sources)for(int i=-8;i<9;++i)
            cleared+=tree.clearance(1.2,q,{s.x+i*.0001,s.y+.0002},.003,0)>.003;
        std::cout<<q<<" "<<cleared<<" "<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<"\n";
    }
}
