cmake_minimum_required(VERSION 3.29)

set(CMAKE_CXX_COMPILER "/usr/bin/mpicxx")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)


project(xpic VERSION 0.1 LANGUAGES CXX)

set(JSON_DIR "external/json")
set(PETSC_DIR "external/petsc")

set(CMAKE_CXX_FLAGS "-std=c++20 -fpic -fopenmp -pthread")

# @todo: should be switched with different CMAKE_BUILD_TYPE
set(PETSC_ARCH
  linux-mpi-debug
  # linux-mpi-opt
)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter -Wno-unused-variable")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ftree-vectorize")

set(main
  src/main.cpp
  src/constants.cpp
)

set(interfaces
  src/interfaces/simulation.cpp
  src/interfaces/point.cpp
  src/interfaces/sort_parameters.cpp
  src/interfaces/particles.cpp
)

set(utils
  src/utils/configuration.cpp
  src/utils/mpi_binary_file.cpp
  src/utils/sync_binary_file.cpp
  src/utils/region_operations.cpp
  src/utils/rotor.cpp
)

set(impls_common
  src/impls/particle_shape.cpp
  src/impls/simple_interpolation.cpp
)

set(impls_basic
  src/impls/basic/simulation.cpp
  src/impls/basic/particles.cpp
  src/impls/basic/diagnostics/field_view.cpp
  src/impls/basic/diagnostics/fields_energy.cpp
  src/impls/basic/diagnostics/distribution_moment.cpp
  src/impls/basic/diagnostics/builders/diagnostic_builder.cpp
  src/impls/basic/diagnostics/builders/field_view_builder.cpp
  src/impls/basic/diagnostics/builders/distribution_moment_builder.cpp
)

set(impls_ricketson
  src/impls/ricketson/simulation.cpp
  src/impls/ricketson/particles.cpp
)

add_executable(
  ${PROJECT_NAME}
  ${main}
  ${interfaces}
  ${utils}
  ${impls_common}
  ${impls_basic}
  ${impls_ricketson}
)

target_precompile_headers(
  ${PROJECT_NAME}
  PRIVATE src/pch.h
)

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE .
  PRIVATE ${JSON_DIR}/include
  PRIVATE ${PETSC_DIR}/include
  PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/include
)

target_link_directories(
  ${PROJECT_NAME}
  PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/lib
)

target_link_libraries(
  ${PROJECT_NAME}
  petsc
  f2clapack
  f2cblas
  m
  X11
)

