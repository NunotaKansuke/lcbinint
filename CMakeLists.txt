cmake_minimum_required(VERSION 3.16)
project(lcbinint LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)

option(LCBININT_BUILD_PYTHON "Build pybind11 Python extension" ON)
option(LCBININT_ENABLE_OPENMP "Parallelize independent parameter rows" ON)
option(LCBININT_ENABLE_JAX_FFI "Build the experimental JAX CPU FFI handler" ON)
option(LCBININT_BUILD_TESTS "Build lcbinint standalone tests" ${BUILD_TESTING})

if(LCBININT_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
endif()

find_package(GSL QUIET)
if(NOT GSL_FOUND)
    set(LCBININT_GSL_HINTS)
    if(DEFINED ENV{GSL_ROOT})
        list(APPEND LCBININT_GSL_HINTS "$ENV{GSL_ROOT}")
    endif()
    list(APPEND LCBININT_GSL_HINTS
        "$ENV{HOME}/local/gsl")
    find_path(GSL_INCLUDE_DIR gsl/gsl_poly.h
        HINTS
            ${LCBININT_GSL_HINTS}
        PATH_SUFFIXES include
        PATHS
            "/usr/include"
            "/usr/local/include")
    find_library(GSL_LIBRARY gsl
        HINTS
            ${LCBININT_GSL_HINTS}
        PATH_SUFFIXES lib lib64
        PATHS
            "/usr/lib64"
            "/usr/lib"
            "/usr/local/lib")
    find_library(GSLCBLAS_LIBRARY gslcblas
        HINTS
            ${LCBININT_GSL_HINTS}
        PATH_SUFFIXES lib lib64
        PATHS
            "/usr/lib64"
            "/usr/lib"
            "/usr/local/lib")
    if(NOT GSL_INCLUDE_DIR OR NOT GSL_LIBRARY OR NOT GSLCBLAS_LIBRARY)
        message(FATAL_ERROR "GSL was not found. Set GSL_ROOT or install GSL development headers/libraries.")
    endif()
    add_library(GSL::gsl UNKNOWN IMPORTED)
    set_target_properties(GSL::gsl PROPERTIES
        IMPORTED_LOCATION "${GSL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GSL_INCLUDE_DIR}")
    add_library(GSL::gslcblas UNKNOWN IMPORTED)
    set_target_properties(GSL::gslcblas PROPERTIES IMPORTED_LOCATION "${GSLCBLAS_LIBRARY}")
endif()

# Leaf layer: finite/point-source magnification kernels + root solvers.
# No dependency on model/, lc/, or obs/ (see .note/session-2026-07-18-final-review.md, A5).
add_library(lcbinint_magnification
    src/lcbinint/magnification/binary_component_certificate.cpp
    src/lcbinint/magnification/finite_source_magnifier.cpp
    src/lcbinint/magnification/point_source_magnifier.cpp
    src/lcbinint/math/polynomial_roots.cpp
    third_party/skowron_gould/SkowronGould.cpp)
target_include_directories(lcbinint_magnification
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/skowron_gould")
target_link_libraries(lcbinint_magnification PUBLIC GSL::gsl GSL::gslcblas m)

# Lens/trajectory model + light-curve/likelihood API; depends on lcbinint_magnification.
add_library(lcbinint_lightcurve
    src/lcbinint/lcbinint.cpp
    src/lcbinint/model/lens_model.cpp
    src/lcbinint/model/lens_parameters.cpp
    src/lcbinint/model/lens_system.cpp
    src/lcbinint/model/orbital_motion.cpp
    src/lcbinint/model/triple_lens_geometry.cpp
    src/lcbinint/model/trajectory.cpp
    src/lcbinint/lc/light_curve.cpp
    src/lcbinint/obs/coordinates.cpp)
target_include_directories(lcbinint_lightcurve
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_definitions(lcbinint_lightcurve PRIVATE
    LCBININT_EARTH_EPHEMERIS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/earth_orbital_parallax_table.txt")
target_link_libraries(lcbinint_lightcurve PUBLIC lcbinint_magnification)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lcbinint_lightcurve PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(lcbinint_lightcurve PUBLIC LCBININT_HAS_OPENMP=1)
endif()

if(LCBININT_BUILD_TESTS)
    add_executable(test_core tests/unit/test_core.cpp)
    target_link_libraries(test_core PRIVATE lcbinint_lightcurve)
    add_test(NAME unit_core COMMAND test_core)
endif()

add_executable(benchmark_point_hex EXCLUDE_FROM_ALL tests/diagnostics/benchmark_point_hex.cpp)
target_link_libraries(benchmark_point_hex PRIVATE lcbinint_lightcurve)

if(LCBININT_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        RESULT_VARIABLE LCBININT_PYBIND11_RESULT
        OUTPUT_VARIABLE LCBININT_PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(LCBININT_PYBIND11_RESULT EQUAL 0)
        list(PREPEND CMAKE_PREFIX_PATH "${LCBININT_PYBIND11_CMAKE_DIR}")
        set(pybind11_DIR "${LCBININT_PYBIND11_CMAKE_DIR}" CACHE PATH
            "pybind11 CMake package directory" FORCE)
    endif()
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(lcbinint_python
        python/lcbinint_pybind.cpp
        python/bind_lc.cpp
        python/bind_obs.cpp
        python/bind_jax_ir.cpp)
    target_link_libraries(lcbinint_python PRIVATE lcbinint_lightcurve)
    if(LCBININT_ENABLE_JAX_FFI)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c
                "import jax.ffi; print(jax.ffi.include_dir())"
            RESULT_VARIABLE LCBININT_JAX_INCLUDE_RESULT
            OUTPUT_VARIABLE LCBININT_JAX_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(LCBININT_JAX_INCLUDE_RESULT EQUAL 0
                AND EXISTS "${LCBININT_JAX_INCLUDE_DIR}/xla/ffi/api/ffi.h")
            target_include_directories(
                lcbinint_python PRIVATE "${LCBININT_JAX_INCLUDE_DIR}")
            target_compile_definitions(
                lcbinint_python PRIVATE LCBININT_HAS_JAX_FFI=1)
            message(STATUS "Building JAX CPU FFI handler")
        else()
            message(FATAL_ERROR
                "JAX headers were not found while LCBININT_ENABLE_JAX_FFI is ON. "
                "Install jax in the build environment or configure "
                "-DLCBININT_ENABLE_JAX_FFI=OFF explicitly.")
        endif()
    endif()
    set_target_properties(lcbinint_python PROPERTIES OUTPUT_NAME _lcbinint)
    # Copy Python package files to build dir so `import lcbinint` works from build/
    add_custom_command(TARGET lcbinint_python POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/python/lcbinint
            ${CMAKE_CURRENT_BINARY_DIR}/lcbinint
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/python/lcbinint_jax
            ${CMAKE_CURRENT_BINARY_DIR}/lcbinint_jax
        COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/lcbinint_jax/data
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data/earth_orbital_parallax_table.txt
            ${CMAKE_CURRENT_BINARY_DIR}/lcbinint_jax/data/earth_orbital_parallax_table.txt
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:lcbinint_python>
            ${CMAKE_CURRENT_BINARY_DIR}/lcbinint/$<TARGET_FILE_NAME:lcbinint_python>
        COMMENT "Copying lcbinint Python packages to build dir")
    get_filename_component(LCBININT_GSL_LIBRARY_DIR "${GSL_LIBRARY}" DIRECTORY)
    if(LCBININT_GSL_LIBRARY_DIR)
        set_target_properties(lcbinint_python PROPERTIES
            BUILD_RPATH "${LCBININT_GSL_LIBRARY_DIR}"
            INSTALL_RPATH "${LCBININT_GSL_LIBRARY_DIR}")
    endif()
    install(TARGETS lcbinint_python LIBRARY DESTINATION lcbinint)
else()
    message(STATUS "Python extension disabled")
endif()

install(TARGETS lcbinint_magnification lcbinint_lightcurve)
install(DIRECTORY include/lcbinint DESTINATION include)
install(FILES data/earth_orbital_parallax_table.txt DESTINATION share/lcbinint/data)
# Install data beside the package too: importlib.resources then works from a
# wheel without relying on a source-checkout-relative path.
install(FILES data/earth_orbital_parallax_table.txt DESTINATION lcbinint_jax/data)
