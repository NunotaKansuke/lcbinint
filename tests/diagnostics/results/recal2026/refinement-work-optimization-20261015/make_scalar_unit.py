from pathlib import Path
import re
P=Path(__file__).resolve().parent;R=Path('/rogue1_8/nunota/lcbinint');rel='src/lcbinint/magnification/finite_source_magnifier.cpp'
def function(p,name):
 s=p.read_text();m=re.search(r'double source_surface_brightness\([^\n]*\)\n\{.*?\n\}',s,re.S);assert m;return m[0].replace('source_surface_brightness',name)
s='''#include <algorithm>
#include <cmath>
#include <random>
#include <cstring>
#include <iostream>
#include <cstdint>
struct FiniteSourceSettings {double limb_darkening_c,limb_darkening_d;};
'''+function(P/'baseline/source'/rel,'before')+'\n'+function(R/rel,'after')+'''
std::uint64_t bits(double v){std::uint64_t u;std::memcpy(&u,&v,8);return u;}
int main(){std::mt19937_64 rng(20261015);std::uniform_real_distribution<double> u(0,1);int brightness_fail=0,angle_fail=0;constexpr double pi=3.14159265358979323846;for(int i=0;i<1000000;++i){double x=1.2*u(rng)-.1;FiniteSourceSettings f{u(rng),i%4==0?.2:(i%2?-0.:0.)};if(bits(before(x,f))!=bits(after(x,f)))++brightness_fail;double a=2*pi*u(rng),b=2*pi*u(rng),d=a-b;double old=std::remainder(d,2*pi),now=d>pi?d-2*pi:(d<-pi?d+2*pi:d);if(bits(old)!=bits(now))++angle_fail;}std::cout<<"cases=1000000 brightness_bit_mismatches="<<brightness_fail<<" angle_bit_mismatches="<<angle_fail<<'\\n';return brightness_fail||angle_fail;}
''';(P/'check_scalar.cpp').write_text(s)
