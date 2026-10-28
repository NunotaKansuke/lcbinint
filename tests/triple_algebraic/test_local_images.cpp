#include "lcbinint/magnification/triple_approximation.hpp"
#include <iostream>
#include <stdexcept>
using namespace lcbinint;
using namespace lcbinint::magnification;

void check(const model::TripleLensGeometry& geometry,SourcePosition source) {
    PointSourceMagnifier solver;
    const auto& images=solver.triple_image_candidates_cached(geometry,source);
    int count=0,local=0;bool projection_loses_residual=false;
    for(const auto& image:images)if(image.physical) {
        ++count;
        std::complex<long double> residual;
        if(image.local_origin>=0) {
            ++local;const auto pole=geometry.lens_positions[image.local_origin];
            residual={static_cast<long double>(pole.x)-source.x+image.local_offset.x,
                      static_cast<long double>(pole.y)-source.y+image.local_offset.y};
        } else residual={static_cast<long double>(image.position.x)-source.x,
                         static_cast<long double>(image.position.y)-source.y};
        for(std::size_t j=0;j<3;++j) {
            // The production field accessor must preserve the represented
            // displacement, including when the projected image equals a pole.
            const auto d=triple_image_separation(geometry,image,j);
            const std::complex<long double> z(d.real(),d.imag());
            residual-=static_cast<long double>(geometry.masses[j])/std::conj(z);
        }
        if(!(std::abs(residual)<=1e-7L) || !(image.residual<=1e-7))
            throw std::runtime_error("represented lens-equation residual");
        if(image.local_origin>=0) {
            const auto mapped=solver.triple_lens_equation(geometry,image.position);
            const double error=std::hypot(mapped.x-source.x,mapped.y-source.y);
            projection_loses_residual=projection_loses_residual||!std::isfinite(error)||error>1e-7;
        }
    }
    if(count!=4 || local==0 || !projection_loses_residual)
        throw std::runtime_error("missing pole-local root witness");
    const auto packet=triple_approximation::point<true>(solver,geometry,source);
    const auto safety=solver.triple_safety_diagnostic(geometry,source);
    if(!packet.valid || !packet.gradient_valid || packet.images!=4 ||
       !std::isfinite(safety.quadrupole_indicator))
        throw std::runtime_error("local field transport");
    const auto value=[](const auto& g,SourcePosition s) {
        const auto r=PointSourceMagnifier().triple_mag0(g,s);
        if(r.image_count!=4 || !std::isfinite(r.magnification))
            throw std::runtime_error("perturbed point root count");
        return r.magnification;
    };
    for(int parameter=0;parameter<11;++parameter) {
        if(parameter==2)continue; // Point-source rho derivative is identically zero.
        constexpr double h=1e-6;
        auto a=geometry,b=geometry;auto sa=source,sb=source;
        if(parameter<2) {
            (parameter?sa.y:sa.x)+=h;(parameter?sb.y:sb.x)-=h;
        } else if(parameter<5) {
            const int lens=parameter-2;
            if(geometry.masses[lens]<2*h) {
                a.masses[lens]+=h;a.masses[0]-=h;
                b.masses[lens]+=2*h;b.masses[0]-=2*h;
                const double fd=(-3*packet.mu+4*value(a,sa)-value(b,sb))/(2*h);
                if(std::abs(fd-packet.gradient[parameter])>2e-4*(1+std::abs(fd)))
                    throw std::runtime_error("one-sided constrained mass tangent");
                continue;
            }
            a.masses[lens]+=h;a.masses[0]-=h;
            b.masses[lens]-=h;b.masses[0]+=h;
        } else {
            const int lens=(parameter-5)/2;
            if((parameter-5)%2){a.lens_positions[lens].y+=h;b.lens_positions[lens].y-=h;}
            else {a.lens_positions[lens].x+=h;b.lens_positions[lens].x-=h;}
        }
        const double fd=(value(a,sa)-value(b,sb))/(2*h);
        if(std::abs(fd-packet.gradient[parameter])>2e-4*(1+std::abs(fd)))
            throw std::runtime_error("physical point tangent");
    }
}
void check_six(const model::TripleLensGeometry& geometry,SourcePosition source,double reference) {
    PointSourceMagnifier solver;
    for(int warm=0;warm<2;++warm) {
        if(warm)solver.triple_mag0(geometry,{source.x+.001,source.y-.001});
        const auto p=triple_approximation::point<true>(solver,geometry,source,6);
        if(p.images!=6 || !p.valid || !p.gradient_valid || std::abs(p.mu-reference)>2e-10)
            throw std::runtime_error("six-image multistart reference");
        for(int j=0;j<2;++j) {
            auto a=source,b=source;constexpr double h=1e-6;
            (j?a.y:a.x)+=h;(j?b.y:b.x)-=h;
            const auto va=triple_approximation::point<false>(PointSourceMagnifier(),geometry,a,6);
            const auto vb=triple_approximation::point<false>(PointSourceMagnifier(),geometry,b,6);
            const double fd=(va.mu-vb.mu)/(2*h);
            if(va.images!=6 || vb.images!=6 ||
                std::abs(fd-p.gradient[j])>1e-4*(1+std::abs(fd)))
            {std::cerr<<"six-image FD source "<<source.x<<" "<<source.y<<" parameter "<<j
                <<" counts "<<va.images<<" "<<vb.images<<" FD "<<fd<<" analytic "<<p.gradient[j]<<"\n";
                throw std::runtime_error("six-image source tangent");}
        }
    }
}
int main() {
    check(model::make_triple_lens_geometry_vbm(19.75305012937681,15.323735743955856,
        19.051792915401325,.7277460252170198,5.397231219594437e-6),
        {.2026000906468922,-1.7413544639713545});
    check(model::make_triple_lens_geometry_vbm(10.690757151389821,1.436858977965615e-7,
        .30669514889554955,1.9826520680699726,91.0362826939728),
        {-1.532344508428104,.04620698856335079});
    check(model::make_triple_lens_geometry_vbm(100,10000,100.00009999999999,
        3.141592643589793,1e-7),{.0009456856212482125,.0003250826137528465});
    model::TripleLensGeometry tiny;
    tiny.lens_positions={{{0,0},{-1,0},{-100,.123}}};
    tiny.masses={.7,.3,1e-20};
    check(tiny,{.4,.6});
    // Two positive images require four negative images: an even total of
    // four used to hide two lost faint roots in this six-image region.
    {
        const auto g=model::make_triple_lens_geometry_vbm(1.1542319750776786,0.8060114128407783,
            19.209528379785677,2.7028194346270764,2.2570528024188276e-05);
        PointSourceMagnifier solver;int positive=0,negative=0;
        for(const auto& i:solver.triple_image_candidates_cached(g,{-0.06408888124035791,0.10640595325060466}))if(i.physical) {
            if(!(i.residual<=1e-7))throw std::runtime_error("six-image residual");
            if(i.jacobian_determinant>0)++positive;else ++negative;
        }
        if(positive!=2 || negative!=4)throw std::runtime_error("missing negative images");
    }
    check_six(model::make_triple_lens_geometry_vbm(1.2725950334694136,28.452407713501238,0.15575480539239409,-0.30732930862308194,9.810429332179342e-07),{0.49640501183679264,0.11943708241206857},6.444764519040969);
    check_six(model::make_triple_lens_geometry_vbm(1.0,1.0,1.7,1.0471975511965976,1e-07),{0.22354679077699266,-0.0051799935624929905},8.096020451794297);
    std::cout<<"PASS: pole-local roots, original residuals and physical tangents\n";
}
