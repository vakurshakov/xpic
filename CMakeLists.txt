cmake_minimum_required(VERSION 3.10)

set(MPI_DIR $ENV{MPI_DIR})
set(JSON_DIR $ENV{JSON_DIR})
set(PETSC_DIR $ENV{PETSC_DIR})

set(MPI_CXX_COMPILER "${MPI_DIR}/bin/mpicxx")
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

project(xpic VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CONFIGURATION_TYPES
  "Debug"
  "Release"
)

set(CMAKE_CXX_FLAGS "-fpic -fopenmp -Wall -Wextra -Wpedantic -Werror -Wno-missing-field-initializers -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
# `-mno-wbnoinvd` pins the only target feature that `-march=native` detects
# inconsistently across P- and E-cores of hybrid Intel CPUs. Without it a
# parallel build may compile the PCH and the translation units on different
# core types, and GCC rejects the `.gch` as `[-Werror=invalid-pch]`.
# The instruction is ring-0 only, so nothing is lost by disabling it.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -mno-wbnoinvd")

set(PETSC_ARCH_OPT linux-mpi-opt)
set(PETSC_ARCH_DEBUG linux-mpi-debug)

set(interfaces
  src/interfaces/simulation.cpp
  src/interfaces/point.cpp
  src/interfaces/sort_parameters.cpp
  src/interfaces/particles.cpp
  src/interfaces/builder.cpp
  src/interfaces/diagnostic.cpp
)

set(algorithms
  src/algorithms/simple_interpolation.cpp
  src/algorithms/simple_decomposition.cpp
  src/algorithms/esirkepov_decomposition.cpp
  src/algorithms/boris_push.cpp
  src/algorithms/crank_nicolson_push.cpp
  src/algorithms/drift_kinetic_push.cpp
  src/algorithms/implicit_esirkepov.cpp
  src/algorithms/implicit_drift_kinetic.cpp
)

set(diagnostics
  src/diagnostics/field_view.cpp
  src/diagnostics/field_view_zavg.cpp
  src/diagnostics/energy.cpp
  src/diagnostics/distribution_moment.cpp
  src/diagnostics/mat_dump.cpp
  src/diagnostics/log_view.cpp
  src/diagnostics/simulation_backup.cpp
  src/diagnostics/charge_conservation.cpp
  src/diagnostics/momentum_conservation.cpp
  src/diagnostics/velocity_distribution.cpp
  src/diagnostics/table_diagnostic.cpp
  src/diagnostics/builders/diagnostic_builder.cpp
  src/diagnostics/builders/field_view_builder.cpp
  src/diagnostics/builders/field_view_zavg_builder.cpp
  src/diagnostics/builders/distribution_moment_builder.cpp
  src/diagnostics/builders/log_view_builder.cpp
  src/diagnostics/builders/simulation_backup_builder.cpp
  src/diagnostics/builders/velocity_distribution_builder.cpp
)

set(commands
  src/commands/inject_particles.cpp
  src/commands/set_particles.cpp
  src/commands/set_magnetic_field.cpp
  src/commands/remove_particles.cpp
  src/commands/fields_damping.cpp
  src/commands/builders/command_builder.cpp
  src/commands/builders/particles_builder.cpp
  src/commands/builders/inject_particles_builder.cpp
  src/commands/builders/remove_particles_builder.cpp
  src/commands/builders/set_particles_builder.cpp
  src/commands/builders/set_magnetic_field_builder.cpp
  src/commands/builders/fields_damping_builder.cpp
)

set(utils
  src/utils/utils.cpp
  src/utils/configuration.cpp
  src/utils/sync_clock.cpp
  src/utils/operators.cpp
  src/utils/shape.cpp
  src/utils/particles_load.cpp
  src/utils/geometries.cpp
  src/utils/world.cpp
  src/utils/table_function.cpp
  src/utils/ziggurat_gaussian.cpp
)

set(impls
  src/impls/basic/simulation.cpp
  src/impls/basic/particles.cpp
  src/impls/ecsim/simulation.cpp
  src/impls/ecsim/particles.cpp
  src/impls/ecsimcorr/simulation.cpp
  src/impls/ecsimcorr/particles.cpp
  src/impls/eccapfim/simulation.cpp
  src/impls/eccapfim/particles.cpp
  src/impls/drift_kinetic/simulation.cpp
  src/impls/drift_kinetic/diagnostic.cpp
  src/impls/drift_kinetic/particles.cpp
  src/impls/drift_kinetic/segments.cpp
  src/impls/drift_kinetic/inject_particles.cpp
  src/impls/drift_kinetic/set_paired_particles.cpp
)

add_library(
  ${PROJECT_NAME} SHARED
  src/constants.cpp
  ${interfaces}
  ${algorithms}
  ${commands}
  ${diagnostics}
  ${utils}
  ${impls}
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
  PUBLIC ${PETSC_DIR}/$<IF:$<CONFIG:Release>,${PETSC_ARCH_OPT},${PETSC_ARCH_DEBUG}>/include
)

target_link_directories(
  ${PROJECT_NAME}
  PUBLIC ${MPI_DIR}/lib
  PUBLIC ${PETSC_DIR}/$<IF:$<CONFIG:Release>,${PETSC_ARCH_OPT},${PETSC_ARCH_DEBUG}>/lib
)

target_link_libraries(
  ${PROJECT_NAME}
  petsc
  m
)

add_executable(${PROJECT_NAME}.out src/main.cpp)
target_link_libraries(${PROJECT_NAME}.out ${PROJECT_NAME})

enable_testing()
add_subdirectory(tests)
