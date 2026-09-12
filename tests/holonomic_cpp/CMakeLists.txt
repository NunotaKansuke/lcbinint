# ATPT holonomic solver (M7) -- ISOLATED build (design decision 19).
#
# This is a *standalone* CMake project, deliberately NOT add_subdirectory'd
# from the top-level CMakeLists.txt.  A normal `cmake` of the main tree is
# byte-for-byte unaffected by M7.  Configure this one explicitly:
#
#   cmake -S tests/holonomic_cpp -B build-holonomic-m7 \
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-holonomic-m7
#
# Targets:
#   test_holonomic_m7          pure-C++ correctness harness vs the Python
#                              reference TSV (evidence/holonomic/m7_reference.tsv)
#   bench_holonomic_m7         timing harness vs baseline_M0 (needs a cases TSV)
#   _lcbinint_holonomic_m7     optional pybind module, SEPARATELY NAMED --
#                              never the shared _lcbinint.so
#
# The header-only solver lives in src/lcbinint/magnification/holonomic/.
# Nothing here links liblcbinint or rebuilds it.

cmake_minimum_required(VERSION 3.16)
project(lcbinint_holonomic_m7 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(_SRC "${_REPO_ROOT}/src")

# Match the tuned flags the M7 checkpoint benchmarks were taken with.
set(_M7_FLAGS -O3 -march=native -funroll-loops -ffp-contract=fast -fno-math-errno)

add_executable(test_holonomic_m7 "${CMAKE_CURRENT_SOURCE_DIR}/test_holonomic_m7.cpp")
target_include_directories(test_holonomic_m7 PRIVATE "${_SRC}")
target_compile_options(test_holonomic_m7 PRIVATE ${_M7_FLAGS})
target_link_libraries(test_holonomic_m7 PRIVATE quadmath)

add_executable(bench_holonomic_m7 "${CMAKE_CURRENT_SOURCE_DIR}/bench_holonomic_m7.cpp")
target_include_directories(bench_holonomic_m7 PRIVATE "${_SRC}")
target_compile_options(bench_holonomic_m7 PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_holonomic_m7 PRIVATE quadmath)

# Checkpoint revision 2, section 13: V0/V1/V2 three-variant micro-benchmark.
add_executable(bench_v0v1v2 "${CMAKE_CURRENT_SOURCE_DIR}/bench_v0v1v2.cpp")
target_include_directories(bench_v0v1v2 PRIVATE "${_SRC}")
target_compile_options(bench_v0v1v2 PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_v0v1v2 PRIVATE quadmath)

# Intermediate experiment E1: planner cost-split (D14 vs grid probes vs
# quartic_topology vs radial pass) -- unified-engine implementation order.
add_executable(bench_planner_split "${CMAKE_CURRENT_SOURCE_DIR}/bench_planner_split.cpp")
target_include_directories(bench_planner_split PRIVATE "${_SRC}")
target_compile_options(bench_planner_split PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_planner_split PRIVATE quadmath)

# Phase A step 1: binary point-source image solver (seeds for the fast planner).
add_executable(test_point_images "${CMAKE_CURRENT_SOURCE_DIR}/test_point_images.cpp")
target_include_directories(test_point_images PRIVATE "${_SRC}")
target_compile_options(test_point_images PRIVATE ${_M7_FLAGS})
target_link_libraries(test_point_images PRIVATE quadmath)
add_test(NAME holonomic_point_images COMMAND test_point_images)

# Certified degree-4 Sturm topology authority and thin-arc regression.
add_executable(test_quartic_sturm
    "${CMAKE_CURRENT_SOURCE_DIR}/test_quartic_sturm.cpp")
target_include_directories(test_quartic_sturm PRIVATE "${_SRC}")
target_compile_options(test_quartic_sturm PRIVATE ${_M7_FLAGS})
target_link_libraries(test_quartic_sturm PRIVATE quadmath)
add_test(NAME holonomic_quartic_sturm COMMAND test_quartic_sturm)

# Phase A step 2: fast seed-anchored band planner vs classify_cells (parity).
add_executable(bench_fast_bands "${CMAKE_CURRENT_SOURCE_DIR}/bench_fast_bands.cpp")
target_include_directories(bench_fast_bands PRIVATE "${_SRC}")
target_compile_options(bench_fast_bands PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_fast_bands PRIVATE quadmath)

# Phase A step 3: fast planner + sec.5.2 local confidence screen -> D14 rate.
add_executable(bench_fast_bands_screen "${CMAKE_CURRENT_SOURCE_DIR}/bench_fast_bands_screen.cpp")
target_include_directories(bench_fast_bands_screen PRIVATE "${_SRC}")
target_compile_options(bench_fast_bands_screen PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_fast_bands_screen PRIVATE quadmath)

# Phase A step 4: fast-planner topology wired into epoch_jacobian (flag-gated);
# full value+5-Jac epoch parity vs classify_cells + decision-20 re-check.
add_executable(bench_fast_topology "${CMAKE_CURRENT_SOURCE_DIR}/bench_fast_topology.cpp")
target_include_directories(bench_fast_topology PRIVATE "${_SRC}")
target_compile_options(bench_fast_topology PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_fast_topology PRIVATE quadmath)

# Phase B step 0: radial_events / D14 cost split -- decides whether to attack
# the D14 solve (compensated Aberth / secular / warm-start) or also the
# __float128 coefficient construction.
add_executable(bench_d14_split "${CMAKE_CURRENT_SOURCE_DIR}/bench_d14_split.cpp")
target_include_directories(bench_d14_split PRIVATE "${_SRC}")
target_compile_options(bench_d14_split PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_d14_split PRIVATE quadmath)

# Phase B step 1: compensated (double-double) D14 polish vs the __float128
# warm polish -- parity (root sets, residual) + A/B timing of solve_d14().
add_executable(bench_d14_compensated "${CMAKE_CURRENT_SOURCE_DIR}/bench_d14_compensated.cpp")
target_include_directories(bench_d14_compensated PRIVATE "${_SRC}")
target_compile_options(bench_d14_compensated PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_d14_compensated PRIVATE quadmath)

# D14 C3/G4/Z3 block-form: exact cancellation-free coefficients + block D/D'
# evaluation in the Aberth polish (default; HOLO_D14_STRUCT_LEGACY=1 reverts).
# Parity + A/B timing.
add_executable(bench_d14_structure "${CMAKE_CURRENT_SOURCE_DIR}/bench_d14_structure.cpp")
target_include_directories(bench_d14_structure PRIVATE "${_SRC}")
target_compile_options(bench_d14_structure PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_d14_structure PRIVATE quadmath)

# Phase C/E: prepared-epoch geometry cache + Phase B2 all-root D14 warm-start.
# Synthesises an N-epoch source track per bench config and runs it cold
# (V0), L1-topology-reuse (V1), and L1+L2-warm-D14 (V2); reports parity vs
# V0, the false-reuse audit, the reuse mix, and per-epoch wall time.
add_executable(bench_holonomic_trajectory "${CMAKE_CURRENT_SOURCE_DIR}/bench_holonomic_trajectory.cpp")
target_include_directories(bench_holonomic_trajectory PRIVATE "${_SRC}")
target_compile_options(bench_holonomic_trajectory PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_holonomic_trajectory PRIVATE quadmath)

# Per-node cost decomposition inside radius_terms -- sizes the angular
# sqrt(phi) sweep (the Gauss-Manin transport target) against the quartic
# build/solve/polish and classify_cells, post B1/B2.
add_executable(bench_radius_terms_split "${CMAKE_CURRENT_SOURCE_DIR}/bench_radius_terms_split.cpp")
target_include_directories(bench_radius_terms_split PRIVATE "${_SRC}")
target_compile_options(bench_radius_terms_split PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_radius_terms_split PRIVATE quadmath)

# Phase C internal mode router (finite_source_binary) -- plumbing parity
# vs epoch_jacobian / epoch_jacobian_prepared over the bench trajectory.
add_executable(test_finite_source_binary "${CMAKE_CURRENT_SOURCE_DIR}/test_finite_source_binary.cpp")
target_include_directories(test_finite_source_binary PRIVATE "${_SRC}")
target_compile_options(test_finite_source_binary PRIVATE ${_M7_FLAGS})
target_link_libraries(test_finite_source_binary PRIVATE quadmath)

# (m,v) root-pair radial transport -- port correctness vs the Python
# reference, boundary_quartic_dR vs finite differences, and a transport
# smoke test (RK4 + Newton corrector) with the tangency fail-closed guard.
add_executable(test_root_pair "${CMAKE_CURRENT_SOURCE_DIR}/test_root_pair.cpp")
target_include_directories(test_root_pair PRIVATE "${_SRC}")
target_compile_options(test_root_pair PRIVATE ${_M7_FLAGS})
target_link_libraries(test_root_pair PRIVATE quadmath)

# Regularized holonomic transport (rescue spikes a/b/c) -- holonomic_transport.hpp
# value vs the 64-pt angular sweep, analytic d/dP_j[v K] vs central FD, and the
# epoch_jacobian grad_mu chain with the transport forced ON.
add_executable(test_holonomic_transport "${CMAKE_CURRENT_SOURCE_DIR}/test_holonomic_transport.cpp")
target_include_directories(test_holonomic_transport PRIVATE "${_SRC}")
target_compile_options(test_holonomic_transport PRIVATE ${_M7_FLAGS})
target_link_libraries(test_holonomic_transport PRIVATE quadmath)

# Phase D value lane: fused kValueJacobian (discard Jac) vs epoch_value /
# epoch_value_prepared through the production router -- wall-clock, mu parity,
# status parity, over {u=0, u=case} x {cold, L2-prepared}.
add_executable(bench_value_lane "${CMAKE_CURRENT_SOURCE_DIR}/bench_value_lane.cpp")
target_include_directories(bench_value_lane PRIVATE "${_SRC}")
target_compile_options(bench_value_lane PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_value_lane PRIVATE quadmath)

# 3-solver whole-epoch benchmark: (V0) current direct / (V1) +(m,v) transport /
# (V2) +(m,v)+regularized holonomic.  Wall-clock median/p90/p95/p99, mu +
# grad_mu parity vs V0, status counts -- decision-20 gate.
add_executable(bench_holonomic_3solver "${CMAKE_CURRENT_SOURCE_DIR}/bench_holonomic_3solver.cpp")
target_include_directories(bench_holonomic_3solver PRIVATE "${_SRC}")
target_compile_options(bench_holonomic_3solver PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_holonomic_3solver PRIVATE quadmath)

# Full coupled-state ODE transport benchmark: (V0) direct / (V1) +(m,v) /
# (V2) +semi-holonomic per-node v*K / (V3) +full coupled-state ODE march.
# value-only and value+5-Jac lanes timed separately; whole-epoch + radial-pass
# percentiles; accuracy + status + fallback rate; ends with a full-ODE audit
# (angular_sweep_nodes == 0, per_node_quartic_solves == 0 on the V3 path).
add_executable(bench_holonomic_ode "${CMAKE_CURRENT_SOURCE_DIR}/bench_holonomic_ode.cpp")
target_include_directories(bench_holonomic_ode PRIVATE "${_SRC}")
target_compile_options(bench_holonomic_ode PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_holonomic_ode PRIVATE quadmath)

# Full coupled-state ODE transport correctness + full-ODE audit (fast; ctest).
add_executable(test_holonomic_ode "${CMAKE_CURRENT_SOURCE_DIR}/test_holonomic_ode.cpp")
target_include_directories(test_holonomic_ode PRIVATE "${_SRC}")
target_compile_options(test_holonomic_ode PRIVATE ${_M7_FLAGS})
target_link_libraries(test_holonomic_ode PRIVATE quadmath)

# Equation-derived Gauss--Manin connection and true Taylor recurrence.
# This target is independent of the existing K-rule / Chebyshev packet.
add_executable(test_gm_taylor "${CMAKE_CURRENT_SOURCE_DIR}/test_gm_taylor.cpp")
target_include_directories(test_gm_taylor PRIVATE "${_SRC}")
target_compile_options(test_gm_taylor PRIVATE ${_M7_FLAGS})
target_link_libraries(test_gm_taylor PRIVATE quadmath)

add_executable(test_gm_physical "${CMAKE_CURRENT_SOURCE_DIR}/test_gm_physical.cpp")
target_include_directories(test_gm_physical PRIVATE "${_SRC}")
target_compile_options(test_gm_physical PRIVATE ${_M7_FLAGS})
target_link_libraries(test_gm_physical PRIVATE quadmath)

add_executable(bench_gm_taylor "${CMAKE_CURRENT_SOURCE_DIR}/bench_gm_taylor.cpp")
target_include_directories(bench_gm_taylor PRIVATE "${_SRC}")
target_compile_options(bench_gm_taylor PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_gm_taylor PRIVATE quadmath)

add_executable(bench_gm_e2e "${CMAKE_CURRENT_SOURCE_DIR}/bench_gm_e2e.cpp")
target_include_directories(bench_gm_e2e PRIVATE "${_SRC}")
target_compile_options(bench_gm_e2e PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_gm_e2e PRIVATE quadmath)

add_executable(test_chart_p4 "${CMAKE_CURRENT_SOURCE_DIR}/test_chart_p4.cpp")
target_include_directories(test_chart_p4 PRIVATE "${_SRC}")
target_compile_options(test_chart_p4 PRIVATE ${_M7_FLAGS})
target_link_libraries(test_chart_p4 PRIVATE quadmath)

add_executable(bench_chart_p4 "${CMAKE_CURRENT_SOURCE_DIR}/bench_chart_p4.cpp")
target_include_directories(bench_chart_p4 PRIVATE "${_SRC}")
target_compile_options(bench_chart_p4 PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_chart_p4 PRIVATE quadmath)

enable_testing()
add_test(NAME holonomic_m7_reference
         COMMAND test_holonomic_m7
                 "${_REPO_ROOT}/evidence/holonomic/m7_reference.tsv")
add_test(NAME holonomic_transport COMMAND test_holonomic_transport)
add_test(NAME holonomic_finite_source_binary
         COMMAND test_finite_source_binary "/tmp/bench_cases.tsv")
add_test(NAME holonomic_root_pair
         COMMAND test_root_pair
                 "${_REPO_ROOT}/evidence/holonomic/root_pair_ref.tsv"
                 "/tmp/bench_cases.tsv")
add_test(NAME holonomic_ode
         COMMAND test_holonomic_ode "/tmp/bench_cases.tsv")
add_test(NAME holonomic_gm_taylor COMMAND test_gm_taylor)
add_test(NAME holonomic_gm_physical COMMAND test_gm_physical)
add_test(NAME holonomic_chart_p4 COMMAND test_chart_p4 "/tmp/bench_cases.tsv")

add_executable(test_reciprocal_chart
    "${CMAKE_CURRENT_SOURCE_DIR}/test_reciprocal_chart.cpp")
target_include_directories(test_reciprocal_chart PRIVATE "${_SRC}")
target_compile_options(test_reciprocal_chart PRIVATE ${_M7_FLAGS})
target_link_libraries(test_reciprocal_chart PRIVATE quadmath)
add_test(NAME holonomic_reciprocal_chart
    COMMAND test_reciprocal_chart
    "${_REPO_ROOT}/evidence/holonomic/gm_coverage_cases.tsv")

# --- optional Python binding (separately named, isolated) ------------------
option(LCBININT_BUILD_HOLONOMIC_M7_PY "Build the isolated _lcbinint_holonomic_m7 pybind module" ON)
if(LCBININT_BUILD_HOLONOMIC_M7_PY)
    find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
    if(Python3_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
            OUTPUT_VARIABLE _PB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _PB_RC)
        if(_PB_RC EQUAL 0)
            set(pybind11_DIR "${_PB_DIR}" CACHE PATH "" FORCE)
            find_package(pybind11 CONFIG QUIET)
        endif()
    endif()
    if(pybind11_FOUND)
        pybind11_add_module(_lcbinint_holonomic_m7
            "${_REPO_ROOT}/python/bind_holonomic.cpp")
        target_include_directories(_lcbinint_holonomic_m7 PRIVATE "${_SRC}")
        target_compile_options(_lcbinint_holonomic_m7 PRIVATE ${_M7_FLAGS})
        target_link_libraries(_lcbinint_holonomic_m7 PRIVATE quadmath)
    else()
        message(STATUS "pybind11 not found -- skipping _lcbinint_holonomic_m7 "
                       "(C++ harness still builds)")
    endif()
endif()

add_executable(bench_gm_coverage "${CMAKE_CURRENT_SOURCE_DIR}/bench_gm_coverage.cpp")
target_include_directories(bench_gm_coverage PRIVATE "${_SRC}")
target_compile_options(bench_gm_coverage PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_gm_coverage PRIVATE quadmath)
add_executable(test_gm_coverage "${CMAKE_CURRENT_SOURCE_DIR}/test_gm_coverage.cpp")
target_include_directories(test_gm_coverage PRIVATE "${_SRC}")
target_compile_options(test_gm_coverage PRIVATE ${_M7_FLAGS})
target_link_libraries(test_gm_coverage PRIVATE quadmath)
add_test(NAME holonomic_gm_coverage COMMAND test_gm_coverage)
add_executable(bench_gm_cold_warm "${CMAKE_CURRENT_SOURCE_DIR}/bench_gm_cold_warm.cpp")
target_include_directories(bench_gm_cold_warm PRIVATE "${_SRC}")
target_compile_options(bench_gm_cold_warm PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_gm_cold_warm PRIVATE quadmath)
add_executable(bench_gm_angular_jac "${CMAKE_CURRENT_SOURCE_DIR}/bench_gm_angular_jac.cpp")
target_include_directories(bench_gm_angular_jac PRIVATE "${_SRC}")
target_compile_options(bench_gm_angular_jac PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_gm_angular_jac PRIVATE quadmath)
add_executable(bench_gm_connection_coverage "${CMAKE_CURRENT_SOURCE_DIR}/bench_gm_connection_coverage.cpp")
target_include_directories(bench_gm_connection_coverage PRIVATE "${_SRC}")
target_compile_options(bench_gm_connection_coverage PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_gm_connection_coverage PRIVATE quadmath)

add_executable(test_gm_lauricella6 "${CMAKE_CURRENT_SOURCE_DIR}/test_gm_lauricella6.cpp")
target_include_directories(test_gm_lauricella6 PRIVATE "${_SRC}")
target_compile_options(test_gm_lauricella6 PRIVATE ${_M7_FLAGS})
target_link_libraries(test_gm_lauricella6 PRIVATE quadmath)
add_test(NAME holonomic_gm_lauricella6 COMMAND test_gm_lauricella6)
add_executable(bench_gm_lauricella6 "${CMAKE_CURRENT_SOURCE_DIR}/bench_gm_lauricella6.cpp")
target_include_directories(bench_gm_lauricella6 PRIVATE "${_SRC}")
target_compile_options(bench_gm_lauricella6 PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_gm_lauricella6 PRIVATE quadmath)

# Phase 7: shared V2 geometry/D14/root/F0 plan; pure LD evaluator A/B between
# the incumbent 16-node K-rule and the equation-derived PF6/fold packet.
add_executable(bench_gm_phase7_shared
    "${CMAKE_CURRENT_SOURCE_DIR}/bench_gm_phase7_shared.cpp")
target_include_directories(bench_gm_phase7_shared PRIVATE "${_SRC}")
target_compile_options(bench_gm_phase7_shared PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_gm_phase7_shared PRIVATE quadmath)
add_executable(test_gm_phase7_shared
    "${CMAKE_CURRENT_SOURCE_DIR}/test_gm_phase7_shared.cpp")
target_include_directories(test_gm_phase7_shared PRIVATE "${_SRC}")
target_compile_options(test_gm_phase7_shared PRIVATE ${_M7_FLAGS})
target_link_libraries(test_gm_phase7_shared PRIVATE quadmath)
add_test(NAME holonomic_gm_phase7_shared COMMAND test_gm_phase7_shared)

# Actual finite_source_binary production-router profile for current V2.
# This is intentionally separate from the Phase 7 shared-geometry harness.
add_executable(bench_v2_profile
    "${CMAKE_CURRENT_SOURCE_DIR}/bench_v2_profile.cpp")
target_include_directories(bench_v2_profile PRIVATE "${_SRC}")
target_compile_options(bench_v2_profile PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_v2_profile PRIVATE quadmath)

# Opt-in D14 per-root scheduling diagnostics.  This records root work and
# non-overlapping D14/topology stages for cold and trajectory-warm calls.
add_executable(bench_d14_root_work_detail
    "${_REPO_ROOT}/benchmarks/holonomic/bench_d14_root_work_detail.cpp")
target_include_directories(bench_d14_root_work_detail PRIVATE "${_SRC}")
target_compile_options(bench_d14_root_work_detail PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_d14_root_work_detail PRIVATE quadmath)

add_executable(bench_d14_event_contract
    "${_REPO_ROOT}/benchmarks/holonomic/bench_d14_event_contract.cpp")
target_include_directories(bench_d14_event_contract PRIVATE "${_SRC}")
target_compile_options(bench_d14_event_contract PRIVATE ${_M7_FLAGS})
target_compile_definitions(bench_d14_event_contract PRIVATE
    HOLO_D14_EVENT_CONTRACT_RESEARCH=1)
target_link_libraries(bench_d14_event_contract PRIVATE quadmath)

# Uninstrumented V2 whole-epoch value/value+5Jac cold/warm timing.  The
# angular reference is used only for parity; transport overrides force the
# measured route to the current V2 production candidate.
add_executable(bench_v2_whole
    "${CMAKE_CURRENT_SOURCE_DIR}/bench_v2_whole.cpp")
target_include_directories(bench_v2_whole PRIVATE "${_SRC}")
target_compile_options(bench_v2_whole PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_v2_whole PRIVATE quadmath)

# Research-only D14 matrix-polynomial / companion-pencil comparison.  A QZ
# dependency is intentionally not added to the isolated build; the benchmark
# records the exact identity and pencil assembly before any external solver.
add_executable(bench_d14_matrix
    "${CMAKE_CURRENT_SOURCE_DIR}/bench_d14_matrix.cpp")
target_include_directories(bench_d14_matrix PRIVATE "${_SRC}")
target_compile_options(bench_d14_matrix PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_d14_matrix PRIVATE quadmath)

add_executable(test_adaptive_radial test_adaptive_radial.cpp)
target_include_directories(test_adaptive_radial PRIVATE "${_SRC}")
target_compile_options(test_adaptive_radial PRIVATE ${_M7_FLAGS})
target_link_libraries(test_adaptive_radial PRIVATE quadmath)
add_test(NAME holonomic_adaptive_radial COMMAND test_adaptive_radial)

foreach(adaptive_target bench_adaptive_radial check_adaptive_reference check_adaptive_jacobian)
    add_executable(${adaptive_target} ${adaptive_target}.cpp)
    target_include_directories(${adaptive_target} PRIVATE "${_SRC}")
    target_compile_options(${adaptive_target} PRIVATE ${_M7_FLAGS})
    target_link_libraries(${adaptive_target} PRIVATE quadmath)
endforeach()
add_executable(bench_adaptive_contracts bench_adaptive_contracts.cpp)
target_include_directories(bench_adaptive_contracts PRIVATE "${_SRC}")
target_compile_options(bench_adaptive_contracts PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_adaptive_contracts PRIVATE quadmath)
add_executable(bench_adaptive_diagnostics bench_adaptive_diagnostics.cpp)
target_include_directories(bench_adaptive_diagnostics PRIVATE "${_SRC}")
target_compile_options(bench_adaptive_diagnostics PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_adaptive_diagnostics PRIVATE quadmath)
add_executable(bench_adaptive_controls bench_adaptive_controls.cpp)
target_include_directories(bench_adaptive_controls PRIVATE "${_SRC}")
target_compile_options(bench_adaptive_controls PRIVATE ${_M7_FLAGS})
target_link_libraries(bench_adaptive_controls PRIVATE quadmath)
add_executable(check_adaptive_limits check_adaptive_limits.cpp)
target_include_directories(check_adaptive_limits PRIVATE "${_SRC}")
target_compile_options(check_adaptive_limits PRIVATE ${_M7_FLAGS})
target_link_libraries(check_adaptive_limits PRIVATE quadmath)

add_executable(test_d14_positive test_d14_positive.cpp)
target_include_directories(test_d14_positive PRIVATE "${_SRC}")
target_compile_options(test_d14_positive PRIVATE ${_M7_FLAGS})
target_link_libraries(test_d14_positive PRIVATE quadmath)
add_test(NAME holonomic_d14_positive COMMAND test_d14_positive)

add_executable(test_d14_root_scheduling test_d14_root_scheduling.cpp)
target_include_directories(test_d14_root_scheduling PRIVATE "${_SRC}")
target_compile_options(test_d14_root_scheduling PRIVATE ${_M7_FLAGS})
target_link_libraries(test_d14_root_scheduling PRIVATE quadmath)
add_test(NAME holonomic_d14_root_scheduling COMMAND test_d14_root_scheduling)

add_executable(test_radial_atlas test_radial_atlas.cpp)
target_include_directories(test_radial_atlas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_compile_options(test_radial_atlas PRIVATE ${_M7_FLAGS})
target_link_libraries(test_radial_atlas PRIVATE quadmath)
add_test(NAME holonomic_radial_atlas COMMAND test_radial_atlas)
