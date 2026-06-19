cmake_minimum_required(VERSION 3.16)
project(lcbinint LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)

option(LCBININT_BUILD_PYTHON "Build pybind11 Python extension" ON)

find_package(GSL QUIET)
if(NOT GSL_FOUND)
    set(LCBININT_GSL_HINTS)
    if(DEFINED ENV{GSL_ROOT})
        list(APPEND LCBININT_GSL_HINTS "$ENV{GSL_ROOT}")
    endif()
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

add_library(lcbinint_core
    src/lcbinint/lcbinint.cpp
    src/lcbinint/magnification/point_source_magnifier.cpp
    src/lcbinint/math/polynomial_roots.cpp
    src/lcbinint/model/lens_model.cpp
    src/lcbinint/model/lens_parameters.cpp
    src/lcbinint/model/lens_system.cpp
    src/lcbinint/model/trajectory.cpp
    third_party/skowron_gould/SkowronGould.cpp)
target_include_directories(lcbinint_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/skowron_gould")
target_link_libraries(lcbinint_core PUBLIC GSL::gsl GSL::gslcblas m)

add_executable(test_core tests/unit/test_core.cpp)
target_link_libraries(test_core PRIVATE lcbinint_core)
add_test(NAME unit_core COMMAND test_core)

if(LCBININT_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(lcbinint_python python/lcbinint_pybind.cpp)
    target_link_libraries(lcbinint_python PRIVATE lcbinint_core)
    set_target_properties(lcbinint_python PROPERTIES OUTPUT_NAME lcbinint)
    install(TARGETS lcbinint_python LIBRARY DESTINATION .)
else()
    message(STATUS "Python extension disabled")
endif()

install(TARGETS lcbinint_core)
install(DIRECTORY include/lcbinint DESTINATION include)
