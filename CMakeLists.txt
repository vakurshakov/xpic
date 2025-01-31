cmake_minimum_required(VERSION 3.10)

set(MPI_DIR $ENV{MPI_DIR})
set(JSON_DIR $ENV{JSON_DIR})
set(PETSC_DIR $ENV{PETSC_DIR})

set(MPI_CXX_COMPILER "${MPI_DIR}/bin/mpicxx")
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

if(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
  set(PETSC_ARCH linux-mpi-opt)
else()
  set(PETSC_ARCH linux-mpi-debug)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# enables clang-tidy in the `build` directory
set(CMAKE_EXPORT_COMPILE_COMMANDS true)

project(xpic VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-fpic -fopenmp -pthread")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -Wall -Wextra -Wpedantic -Werror -Wno-missing-field-initializers")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")

set(interfaces
  src/interfaces/simulation.cpp
  src/interfaces/point.cpp
  src/interfaces/sort_parameters.cpp
  src/interfaces/particles.cpp
  src/interfaces/world.cpp
  src/interfaces/builder.cpp
)

set(algorithms
  src/algorithms/simple_interpolation.cpp
  src/algorithms/simple_decomposition.cpp
  src/algorithms/esirkepov_decomposition.cpp
  src/algorithms/boris_push.cpp
)

set(diagnostics
  src/diagnostics/field_view.cpp
  src/diagnostics/fields_energy.cpp
  src/diagnostics/particles_energy.cpp
  src/diagnostics/distribution_moment.cpp
  src/diagnostics/mat_dump.cpp
  src/diagnostics/builders/diagnostic_builder.cpp
  src/diagnostics/builders/field_view_builder.cpp
  src/diagnostics/builders/distribution_moment_builder.cpp
)

set(commands
  src/commands/inject_particles.cpp
  src/commands/set_particles.cpp
  src/commands/setup_magnetic_field.cpp
  src/commands/remove_particles.cpp
  src/commands/fields_damping.cpp
  src/commands/builders/command_builder.cpp
  src/commands/builders/particles_builder.cpp
  src/commands/builders/inject_particles_builder.cpp
  src/commands/builders/remove_particles_builder.cpp
  src/commands/builders/set_particles_builder.cpp
  src/commands/builders/setup_magnetic_field_builder.cpp
  src/commands/builders/fields_damping_builder.cpp
)

set(utils
  src/utils/configuration.cpp
  src/utils/mpi_binary_file.cpp
  src/utils/sync_file.cpp
  src/utils/sync_binary_file.cpp
  src/utils/region_operations.cpp
  src/utils/operators.cpp
  src/utils/shape.cpp
  src/utils/particles_load.cpp
  src/utils/geometries.cpp
)

set(impls_basic
  src/impls/basic/simulation.cpp
  src/impls/basic/particles.cpp
)

set(impls_ecsimcorr
  src/impls/ecsimcorr/simulation.cpp
  src/impls/ecsimcorr/particles.cpp
  src/impls/ecsimcorr/energy_conservation.cpp
  src/impls/ecsimcorr/charge_conservation.cpp
)

set(impls_ricketson
  src/impls/ricketson/simulation.cpp
  src/impls/ricketson/particles.cpp
)

add_library(
  ${PROJECT_NAME} SHARED
  src/constants.cpp
  ${interfaces}
  ${algorithms}
  ${commands}
  ${diagnostics}
  ${utils}
  ${impls_basic}
  ${impls_ecsimcorr}
  ${impls_ricketson}
)

target_precompile_headers(
  ${PROJECT_NAME}
  PUBLIC src/pch.h
)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC .
  PUBLIC ${MPI_DIR}/include
  PUBLIC ${JSON_DIR}/include
  PUBLIC ${PETSC_DIR}/include
  PUBLIC ${PETSC_DIR}/${PETSC_ARCH}/include
)

target_link_directories(
  ${PROJECT_NAME}
  PUBLIC ${MPI_DIR}/lib
  PUBLIC ${PETSC_DIR}/${PETSC_ARCH}/lib
)

target_link_libraries(
  ${PROJECT_NAME}
  petsc
  f2clapack
  f2cblas
  m
  X11
)

add_executable(${PROJECT_NAME}.out src/main.cpp)
target_link_libraries(${PROJECT_NAME}.out ${PROJECT_NAME})

add_subdirectory(tests)