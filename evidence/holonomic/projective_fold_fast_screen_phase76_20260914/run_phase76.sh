#!/usr/bin/env bash
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
here="$root/evidence/holonomic/projective_fold_fast_screen_phase76_20260914"
input="$here/input_snapshot.tsv"
reference="$here/reference_cases.tsv"
flags=(-O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -I"$root/src"
  -march=native -funroll-loops -ffp-contract=fast -fno-math-errno)
mkdir -p "$here/raw"
[[ "$(awk 'NF && $1 !~ /^#/ { n++ } END { print n+0 }' "$input")" -eq 7216 ]]
[[ "$(awk 'NF && $1 !~ /^#/ { n++ } END { print n+0 }' "$reference")" -eq 110 ]]

{
  printf 'date_utc='; date -u '+%Y-%m-%dT%H:%M:%SZ'
  printf 'git_head='; git -C "$root" rev-parse HEAD
  printf 'input_sha256='; sha256sum "$input"
  printf 'reference_sha256='; sha256sum "$reference"
  printf 'compiler='; c++ --version | head -1
  printf 'cpu='; lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1
  printf 'affinity=taskset -c 0-7; serial runner\n'
  printf 'compile_flags='; printf '%q ' "${flags[@]}" | sed 's/ $//'; printf '\n'
  printf 'arms: baseline=production path; qf=projective fold research with original qf probe; screen=same research path with fail-open interval screen\n'
  printf 'policy=value-only None or ValueFirst+5 analytic Jacobians; mu_atol=1e-16; mu_rtol=1e-3,1e-4; grad_rtol=1e-3; default grad_atol\n'
  printf 'timers=full-cold includes D14/topology+adaptive; full-warm includes L2 warm D14 topology+adaptive; radial-only prebuilds topology outside timer\n'
  printf 'repetitions=3 process repetitions per arm/policy; arm order rotates; each runner excludes its own warmup trajectory\n'
  printf 'unrelated HOLO env vars are unset as in Phase75\n'
} > "$here/manifest.txt"

src="$here/trajectory_ab.cpp"
c++ "${flags[@]}" "$src" -o /tmp/phase76_projective_baseline -lquadmath
c++ "${flags[@]}" -DHOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH \
  "$src" -o /tmp/phase76_projective_qf -lquadmath
c++ "${flags[@]}" -DHOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH \
  -DHOLO_ADAPTIVE_PROJECTIVE_P4_FAST_SCREEN "$src" \
  -o /tmp/phase76_projective_screen -lquadmath
c++ "${flags[@]}" "$here/projective_screen_scan.cpp" \
  -o /tmp/phase76_projective_screen_scan -lquadmath
c++ "${flags[@]}" "$here/projective_trajectory_ab.cpp" \
  -o /tmp/phase76_target_baseline -lquadmath
c++ "${flags[@]}" -DHOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH \
  "$here/projective_trajectory_ab.cpp" -o /tmp/phase76_target_qf -lquadmath
c++ "${flags[@]}" -DHOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH \
  -DHOLO_ADAPTIVE_PROJECTIVE_P4_FAST_SCREEN \
  "$here/projective_trajectory_ab.cpp" -o /tmp/phase76_target_screen -lquadmath
sha256sum /tmp/phase76_projective_{baseline,qf,screen} \
  /tmp/phase76_target_{baseline,qf,screen} \
  /tmp/phase76_projective_screen_scan >> "$here/manifest.txt"
printf '\nsource_sha256:\n' >> "$here/manifest.txt"
sha256sum "$root/src/lcbinint/magnification/holonomic/projective_fold_screen.hpp" \
  "$root/src/lcbinint/magnification/holonomic/adaptive_epoch.hpp" \
  "$root/tests/holonomic_cpp/test_projective_fold.cpp" \
  "$here/trajectory_ab.cpp" "$here/projective_trajectory_ab.cpp" \
  "$here/projective_screen_scan.cpp" "$here/run_phase76.sh" \
  "$here/summarize_phase76.py" >> "$here/manifest.txt"

: > "$here/run.log"
arms=(baseline qf screen)
for policy in value jac; do
  policy_index=0; [[ "$policy" == jac ]] && policy_index=1
  for rep in 0 1 2; do
    shift=$(((rep+policy_index)%3))
    ordered=("${arms[@]:shift}" "${arms[@]:0:shift}")
    for arm in "${ordered[@]}"; do
      binary="/tmp/phase76_projective_${arm}"
      output="$here/raw/${arm}_${policy}_r${rep}.tsv"
      {
        printf '\nrun_start=%s arm=%s policy=%s rep=%s\n' \
          "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$arm" "$policy" "$rep"
        env -u HOLO_D14_EVENT_CONTRACT_ACCEPT -u HOLO_MV_TRANSPORT_LEGACY \
            -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
          taskset -c 0-7 "$binary" "$input" "$output" "$policy" "$rep"
        printf 'run_end=%s arm=%s policy=%s rep=%s rows=' \
          "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$arm" "$policy" "$rep"
        wc -l < "$output"
      } >> "$here/run.log" 2>&1
      [[ "$(wc -l < "$output")" -eq 14434 ]]
      gzip -n -f "$output"
    done
  done
done

env -u HOLO_D14_EVENT_CONTRACT_ACCEPT -u HOLO_MV_TRANSPORT_LEGACY \
    -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
  taskset -c 0-7 /tmp/phase76_projective_screen_scan "$reference" \
  "$here/reference_screen_scan.tsv" reference 2> "$here/reference_screen_scan.log"
env -u HOLO_D14_EVENT_CONTRACT_ACCEPT -u HOLO_MV_TRANSPORT_LEGACY \
    -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
  taskset -c 0-7 /tmp/phase76_projective_screen_scan "$input" \
  "$here/trajectory_screen_scan.tsv" snapshot 2> "$here/trajectory_screen_scan.log"

for arm in baseline qf screen; do
  case "$arm" in
    baseline) binary=/tmp/phase76_target_baseline ;;
    qf) binary=/tmp/phase76_target_qf ;;
    screen) binary=/tmp/phase76_target_screen ;;
  esac
  for policy in value jac; do
    output="$here/raw/targeted_${arm}_${policy}.tsv"
    env -u HOLO_D14_EVENT_CONTRACT_ACCEPT -u HOLO_MV_TRANSPORT_LEGACY \
        -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
      taskset -c 0-7 "$binary" "$output" "$policy" 0 \
      2> "$here/raw/targeted_${arm}_${policy}.log"
    [[ "$(wc -l < "$output")" -eq 2305 ]]
  done
done

python3 "$here/summarize_phase76.py" --root "$here" > "$here/summary.json"
sha256sum "$here/input_snapshot.tsv" "$here/reference_cases.tsv" \
  "$here/manifest.txt" "$here/run_phase76.sh" "$here/trajectory_ab.cpp" \
  "$here/projective_screen_scan.cpp" "$here/projective_trajectory_ab.cpp" \
  "$here/summarize_phase76.py" "$here/reference_screen_scan.tsv" \
  "$here/reference_screen_scan.log" "$here/trajectory_screen_scan.tsv" \
  "$here/trajectory_screen_scan.log" "$here/summary.json" "$here/run.log" \
  "$here"/raw/*.tsv.gz "$here"/raw/targeted_*.tsv "$here"/raw/*.log \
  > "$here/SHA256SUMS"
