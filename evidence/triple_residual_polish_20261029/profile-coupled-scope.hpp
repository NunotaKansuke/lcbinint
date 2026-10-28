#pragma once
#ifdef LCBININT_TRIPLE_COUPLED_PROFILE
#include <array>
#include <time.h>
namespace lcbinint::algebraic_boundary::triple::coupled_profile {
enum Phase { topology, endpoint, chart, pair, angular, witness, count };
inline thread_local std::array<double,count> nanoseconds{},calls{},exclusive{};
inline thread_local int active=-1;inline thread_local double last=0;
enum ChartPhase { discovery, coefficients, moments, chart_count };
inline thread_local std::array<double,chart_count> chart_nanoseconds{},chart_calls{};
inline double cpu_time(){timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return 1e9*double(t.tv_sec)+double(t.tv_nsec);}
struct ChartScope {ChartPhase phase;double start;explicit ChartScope(ChartPhase p):phase(p),start(cpu_time()){}~ChartScope(){chart_nanoseconds[phase]+=cpu_time()-start;chart_calls[phase]+=1;}};
struct Scope {Phase phase;double start;int parent;explicit Scope(Phase p):phase(p),start(cpu_time()),parent(active){if(active>=0)exclusive[active]+=start-last;active=p;last=start;}~Scope(){const double now=cpu_time();exclusive[phase]+=now-last;nanoseconds[phase]+=now-start;calls[phase]+=1;active=parent;last=now;}};
}
#define LCBININT_CHART_SCOPE(name) ::lcbinint::algebraic_boundary::triple::coupled_profile::ChartScope chart_detail_scope(::lcbinint::algebraic_boundary::triple::coupled_profile::name)
#define LCBININT_COUPLED_SCOPE(name) ::lcbinint::algebraic_boundary::triple::coupled_profile::Scope profile_scope(::lcbinint::algebraic_boundary::triple::coupled_profile::name)
#else
#define LCBININT_CHART_SCOPE(name) ((void)0)
#define LCBININT_COUPLED_SCOPE(name) ((void)0)
#endif
