#!/usr/bin/env bash
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
here="$root/evidence/holonomic/projective_fold_final_ab_phase75_20260914"
input="$here/input_snapshot.tsv"
flags=(-O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -I"$root/src"
  -march=native -funroll-loops -ffp-contract=fast -fno-math-errno)
mkdir -p "$here/raw"
[[ "$(awk 'NF && $1 !~ /^#/ { n++ } END { print n+0 }' "$input")" -eq 7216 ]]

{
  printf 'date_utc='; date -u '+%Y-%m-%dT%H:%M:%SZ'
  printf 'git_head='; git -C "$root" rev-parse HEAD
  printf 'input_sha256='; sha256sum "$input"
  printf 'compiler='; c++ --version | head -1
  printf 'cpu='; lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1
  printf 'affinity=taskset -c 0-7; runner is serial'
  printf '\n'
  printf 'baseline_compile=c++ '; printf '%q ' "${flags[@]}" "$here/trajectory_ab.cpp" -o /tmp/phase75_projective_baseline -lquadmath; printf '\n'
  printf 'candidate_compile=c++ '; printf '%q ' "${flags[@]}" -DHOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH "$here/trajectory_ab.cpp" -o /tmp/phase75_projective_candidate -lquadmath; printf '\n'
  printf 'projective_scan_compile=c++ '; printf '%q ' "${flags[@]}" "$here/projective_scan.cpp" -o /tmp/phase75_projective_scan -lquadmath; printf '\n'
  printf 'projective_scan_run=env -u HOLO_D14_EVENT_CONTRACT_ACCEPT -u HOLO_MV_TRANSPORT_LEGACY -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY taskset -c 0-7 /tmp/phase75_projective_scan input_snapshot.tsv projective_scan.tsv\n'
  printf 'policy=adaptive value-only (None) or ValueFirst + 5 analytic Jacobians; mu_atol=1e-16; mu_rtol=1e-3,1e-4; grad_rtol=1e-3; grad_atol=AdaptiveConfig default\n'
  printf 'cold=epoch_adaptive, topology/D14 inside whole timer; warm=prepared_topology L2 D14 roots, topology reuse disabled; first trajectory epoch cold; radial=prebuilt topology outside timer\n'
  printf 'repetitions=3 separate processes per arm/policy, within-repeat arm order alternates; each process has one unmeasured trajectory warm-up per lane\n'
  printf 'HOLO environment variables inherited: '
  if env | rg '^HOLO_' ; then true; else printf '(none)\n'; fi
  printf 'compile_flags='; printf '%q ' "${flags[@]}"; printf '\n'
} > "$here/manifest.txt"

c++ "${flags[@]}" "$here/trajectory_ab.cpp" -o /tmp/phase75_projective_baseline -lquadmath
c++ "${flags[@]}" -DHOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH \
  "$here/trajectory_ab.cpp" -o /tmp/phase75_projective_candidate -lquadmath
sha256sum /tmp/phase75_projective_baseline /tmp/phase75_projective_candidate >> "$here/manifest.txt"

: > "$here/run.log"
for policy in value jac; do
  for rep in 0 1 2; do
    if (( rep % 2 == 0 )); then arms=(baseline candidate); else arms=(candidate baseline); fi
    for arm in "${arms[@]}"; do
      if [[ "$arm" == baseline ]]; then binary=/tmp/phase75_projective_baseline
      else binary=/tmp/phase75_projective_candidate; fi
      output="$here/raw/${arm}_${policy}_r${rep}.tsv"
      {
        printf '\nrun_start=%s arm=%s policy=%s rep=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$arm" "$policy" "$rep"
        env -u HOLO_D14_EVENT_CONTRACT_ACCEPT -u HOLO_MV_TRANSPORT_LEGACY \
            -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
          taskset -c 0-7 "$binary" "$input" "$output" "$policy" "$rep"
        printf 'run_end=%s arm=%s policy=%s rep=%s rows=' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$arm" "$policy" "$rep"
        wc -l < "$output"
      } >> "$here/run.log" 2>&1
      [[ "$(wc -l < "$output")" -eq 14434 ]]
      gzip -n -f "$output"
    done
  done
done

c++ "${flags[@]}" "$here/projective_scan.cpp" -o /tmp/phase75_projective_scan -lquadmath
env -u HOLO_D14_EVENT_CONTRACT_ACCEPT -u HOLO_MV_TRANSPORT_LEGACY \
    -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
  taskset -c 0-7 /tmp/phase75_projective_scan "$input" "$here/projective_scan.tsv" \
  2> "$here/projective_scan.log"
python3 "$here/summarize_phase75.py" --root "$here" > "$here/summary_stdout.json"
cp "$here/summary_stdout.json" "$here/summary.json"
sha256sum "$here/input_snapshot.tsv" "$here/manifest.txt" "$here/run_phase75.sh" \
  "$here/trajectory_ab.cpp" "$here/projective_scan.cpp" "$here/projective_scan.tsv" \
  "$here/projective_scan.log" "$here/summarize_phase75.py" "$here/summary.json" \
  "$here/run.log" "$here"/raw/*.tsv.gz > "$here/SHA256SUMS"
