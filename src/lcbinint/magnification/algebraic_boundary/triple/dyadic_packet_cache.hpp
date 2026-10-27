#pragma once
#include <array>
#include <vector>
namespace lcbinint::algebraic_boundary::triple::packet_cache_detail {
// Canonical birth-order index of a nested dyadic-angle node. All nodes at
// level l occupy a dense prefix of length 2^l-1, independent of final order.
inline unsigned slot(unsigned level,unsigned k){const unsigned shift=__builtin_ctz(k);return (1u<<(level-shift-1))-1+(k>>(shift+1));}
template<class Packet,unsigned Inline=15> class DyadicPacketCache {
 std::array<Packet,Inline> small_;std::vector<Packet> extra_;unsigned completed_=0;
public:
 void prepare(unsigned level){const unsigned count=(1u<<level)-1;if(count>Inline&&extra_.size()<count-Inline)extra_.resize(count-Inline);}
 bool contains(unsigned index)const{return index<completed_;}
 Packet& operator[](unsigned index){return index<Inline?small_[index]:extra_[index-Inline];}
 void finish(unsigned level){completed_=(1u<<level)-1;}
 unsigned completed()const{return completed_;}
};
}
