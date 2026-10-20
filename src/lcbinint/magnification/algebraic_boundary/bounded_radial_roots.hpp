#pragma once

#include "exact_radial_support.hpp"

namespace lcbinint::algebraic_boundary::exact_support_detail {

struct SupportRootBox {
    double lo=0,hi=0;
    int lower_sign=0,upper_sign=0;
};
struct SupportRootResult {
    bool completed=false;
    bool used_bounded=false;
    std::vector<SupportRootBox> boxes;
};

namespace bounded_detail {
using Coefficient=boost::multiprecision::number<boost::multiprecision::cpp_int_backend<
    256,256,boost::multiprecision::signed_magnitude,boost::multiprecision::checked,void>>;
struct CoefficientBox {
    double lo=0,hi=0;
    std::array<Coefficient,15> b;
    unsigned error=0;
    int count=0;
};
struct Result {
    bool completed=false;
    std::vector<CoefficientBox> boxes;
};
inline int sign(const Coefficient& value,unsigned error) {
    const Coefficient bound(error);
    return value>bound?1:value<-bound?-1:value==0 && error==0?0:2;
}
inline int variations(const CoefficientBox& box) {
    int last=0,count=0;
    for(const auto& coefficient:box.b) {
        const int current=sign(coefficient,box.error);
        if(current==2)return -1;
        if(current) {
            count+=last && last!=current;
            last=current;
        }
    }
    return count;
}
inline Coefficient floor_div(const Coefficient& value,int bits) {
    if(!bits)return value;
    const Coefficient mask=(Coefficient(1)<<bits)-1;
    return value<0?-((-value+mask)>>bits):Coefficient(value>>bits);
}
inline std::pair<CoefficientBox,CoefficientBox> split(const CoefficientBox& box) {
    CoefficientBox left,right;
    if(box.error>=4096) {
        left.count=right.count=-1;
        return {std::move(left),std::move(right)};
    }
    const double mid=.5*(box.lo+box.hi);
    left.lo=box.lo;left.hi=right.lo=mid;right.hi=box.hi;
    // Each exact child coefficient is a convex combination of its parents.
    // Their common error therefore stays <= error; flooring adds < one unit.
    left.error=right.error=box.error+1;
    left.b[0]=box.b[0];right.b[14]=box.b[14];
    auto work=box.b;
    for(int k=1;k<=14;++k) {
        for(int j=0;j<=14-k;++j)work[j]+=work[j+1];
        left.b[k]=floor_div(work[0],k);
        right.b[14-k]=floor_div(work[14-k],k);
    }
    left.count=variations(left);right.count=variations(right);
    return {std::move(left),std::move(right)};
}
inline Result isolate_box(const Box& source,int precision=192) {
    Result out;
    if(precision<32 || precision>224)return out;
    int magnitude=0;
    for(const auto& coefficient:source.b)if(coefficient!=0)
        magnitude=std::max(magnitude,int(boost::multiprecision::msb(
            coefficient<0?-coefficient:coefficient))+1);
    const int drop=std::max(0,magnitude-precision);
    CoefficientBox first;
    first.lo=source.lo;first.hi=source.hi;
    for(int i=0;i<=14;++i) {
        const auto& value=source.b[i];
        const bool negative=value<0;
        const Integer absolute=negative?-value:value;
        const Integer center=absolute>>drop;
        first.error|=(center<<drop)!=absolute;
        first.b[i]=negative?-Coefficient(center):Coefficient(center);
    }
    // At most precision bits initially, plus 14 summation bits and the small
    // accumulated error. Even precision=224 stays below the checked 256 bits.
    first.count=variations(first);
    const double minimum=first.lo,maximum=first.hi;
    std::vector<CoefficientBox> pending;
    pending.push_back(std::move(first));
    int nodes=0;
    while(!pending.empty()) {
        if(++nodes>2048)return out;
        auto box=std::move(pending.back());pending.pop_back();
        if(box.count<0 || sign(box.b[0],box.error)==0 ||
           sign(box.b[14],box.error)==0)return out;
        if(!box.count)continue;
        if(box.count==1) {
            out.boxes.push_back(std::move(box));
            continue;
        }
        const double mid=.5*(box.lo+box.hi);
        if(!(mid>box.lo && mid<box.hi))return out;
        auto children=split(box);
        pending.push_back(std::move(children.second));
        pending.push_back(std::move(children.first));
    }
    std::sort(out.boxes.begin(),out.boxes.end(),
        [](const CoefficientBox& a,const CoefficientBox& b){return a.lo<b.lo;});
    auto refine=[](CoefficientBox& box) {
        const double mid=.5*(box.lo+box.hi);
        if(!(mid>box.lo && mid<box.hi))return false;
        auto children=split(box);
        if(children.first.count<0 || children.second.count<0 ||
           children.first.count+children.second.count!=1)return false;
        if(children.first.count==1)box=std::move(children.first);
        else box=std::move(children.second);
        return true;
    };
    for(std::size_t i=1;i<out.boxes.size();++i)
        for(int k=0;out.boxes[i-1].hi>=out.boxes[i].lo;++k)
            if(k==64 || !refine(out.boxes[i-1]) || !refine(out.boxes[i]))return out;
    for(auto& box:out.boxes)
        for(int k=0;box.lo==minimum || box.hi==maximum;++k)
            if(k==64 || !refine(box))return out;
    for(const auto& box:out.boxes)
        if(sign(box.b[0],box.error)==2 || sign(box.b[14],box.error)==2)return out;
    out.completed=true;
    return out;
}
} // namespace bounded_detail

inline SupportRootResult radial_support_roots(const PrimaryFrame& pf) {
    SupportRootResult out;
    Box source;
    if(!make_radial_bernstein_box(pf,source))return out;
    auto bounded=bounded_detail::isolate_box(source);
    if(bounded.completed) {
        out.completed=out.used_bounded=true;
        for(const auto& box:bounded.boxes)
            out.boxes.push_back({box.lo,box.hi,bounded_detail::sign(box.b[0],box.error),
                bounded_detail::sign(box.b[14],box.error)});
        return out;
    }
    // Reuse the exact coefficients already formed; uncertainty never accepts
    // a sign, and does not require constructing the polynomial a second time.
    auto exact=isolate_bernstein(std::move(source));
    out.completed=exact.completed;
    for(const auto& box:exact.boxes)
        out.boxes.push_back({box.lo,box.hi,box.b[0]>0?1:box.b[0]<0?-1:0,
            box.b[14]>0?1:box.b[14]<0?-1:0});
    return out;
}

} // namespace lcbinint::algebraic_boundary::exact_support_detail
