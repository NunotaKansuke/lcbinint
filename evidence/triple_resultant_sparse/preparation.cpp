#include "lcbinint/magnification/algebraic_boundary/triple/support_resultant.hpp"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace lcbinint::algebraic_boundary::triple;
volatile long double sink;
int main(){std::vector<double>times;for(int k=0;k<64;++k){Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};p.by+=k*.00001;auto t=std::chrono::steady_clock::now();resultant_detail::MatrixBasis basis(p);sink=basis.coefficients[1][0][0][0].value;times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()*1000);}std::sort(times.begin(),times.end());std::cout<<"{\"median_preparation_ms\":"<<times[32]<<"}\n";}
