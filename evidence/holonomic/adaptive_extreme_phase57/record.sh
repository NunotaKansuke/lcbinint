set -eu
perf record -e cycles:u -F 499 --call-graph dwarf,8192 -o /tmp/p57.perf.data -- taskset -c 0 /tmp/p57_whole evidence/holonomic/adaptive_extreme_phase56/input_snapshot.tsv /tmp/p57_whole.tsv 20
perf report -i /tmp/p57.perf.data --stdio --no-children --sort symbol --percent-limit 1
perf report -i /tmp/p57.perf.data --stdio --no-children -g none --sort symbol --percent-limit 0.5 --field-separator ';'
