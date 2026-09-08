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

enable_testing()
add_test(NAME holonomic_m7_reference
         COMMAND test_holonomic_m7
                 "${_REPO_ROOT}/evidence/holonomic/m7_reference.tsv")
add_test(NAME holonomic_finite_source_binary
         COMMAND test_finite_source_binary "/tmp/bench_cases.tsv")

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
