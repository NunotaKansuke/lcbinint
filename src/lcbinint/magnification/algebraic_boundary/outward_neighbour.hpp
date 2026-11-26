#pragma once
#include <cstdint>
#include <cstring>
#include <limits>
namespace lcbinint::algebraic_boundary::rounding {
// The same outward neighbours as nextafter, without libm calls in every
// binary64 primitive. No arithmetic is performed on a NaN representation.
inline double neighbour(double x,bool upward) {
    static_assert(sizeof(double)==sizeof(std::uint64_t) &&
                  std::numeric_limits<double>::is_iec559,"requires IEEE binary64");
    std::uint64_t u;std::memcpy(&u,&x,sizeof(u));
    const auto mag=u&UINT64_C(0x7fffffffffffffff);
    if(mag>UINT64_C(0x7ff0000000000000))return x;
    if(u==(upward?UINT64_C(0x7ff0000000000000):UINT64_C(0xfff0000000000000)))return x;
    if(mag==0)u=upward?UINT64_C(1):UINT64_C(0x8000000000000001);
    else if(bool(u>>63)==upward)--u;
    else ++u;
    std::memcpy(&x,&u,sizeof(x));return x;
}
inline __float128 neighbour(__float128 x,bool upward) {
    static_assert(sizeof(__float128)==16 && __FLT128_MANT_DIG__==113,
                  "requires IEEE binary128");
    using U=unsigned __int128;
    constexpr U sign=U(1)<<127,inf=U(0x7fff)<<112;
    U u;std::memcpy(&u,&x,sizeof(u));const U magnitude=u&~sign;
    if(magnitude>inf || u==(upward?inf:sign|inf))return x;
    if(magnitude==0)u=upward?U(1):sign|U(1);
    else if(bool(u&sign)==upward)--u;
    else ++u;
    std::memcpy(&x,&u,sizeof(x));return x;
}
}
