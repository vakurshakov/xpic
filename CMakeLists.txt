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

set(CMAKE_CXX_FLAGS "-fpic -fopenmp -Wall -Wextra -Wpedantic -Werror -Wno-missing-field-initializers")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")

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
  src/algorithms/implicit_esirkepov.cpp
)

set(diagnostics
  src/diagnostics/field_view.cpp
  src/diagnostics/fields_energy.cpp
  src/diagnostics/particles_energy.cpp
  src/diagnostics/distribution_moment.cpp
  src/diagnostics/mat_dump.cpp
  src/diagnostics/log_view.cpp
  src/diagnostics/simulation_backup.cpp
  src/diagnostics/energy_conservation.cpp
  src/diagnostics/charge_conservation.cpp
  src/diagnostics/velocity_distribution.cpp
  src/diagnostics/utils/table_diagnostic.cpp
  src/diagnostics/builders/diagnostic_builder.cpp
  src/diagnostics/builders/field_view_builder.cpp
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
  src/commands/kotelnikov_equilibrium/cyl_without_phi/set_cyl_without_phi.cpp
  src/commands/kotelnikov_equilibrium/cyl_without_phi/table_function.cpp
  src/commands/kotelnikov_equilibrium/cyl_without_phi/ziggurat_gaussian.cpp
  src/commands/builders/command_builder.cpp
  src/commands/builders/particles_builder.cpp
  src/commands/builders/inject_particles_builder.cpp
  src/commands/builders/remove_particles_builder.cpp
  src/commands/builders/set_particles_builder.cpp
  src/commands/builders/set_magnetic_field_builder.cpp
  src/commands/builders/fields_damping_builder.cpp
  src/commands/builders/set_kotelnikov_equilibrium.cpp
)

set(utils
  src/utils/utils.cpp
  src/utils/configuration.cpp
  src/utils/mpi_binary_file.cpp
  src/utils/sync_file.cpp
  src/utils/sync_binary_file.cpp
  src/utils/sync_clock.cpp
  src/utils/operators.cpp
  src/utils/shape.cpp
  src/utils/particles_load.cpp
  src/utils/geometries.cpp
  src/utils/world.cpp
)

set(impls_basic
  src/impls/basic/simulation.cpp
  src/impls/basic/particles.cpp
)

set(impls_ecsim
  src/impls/ecsim/simulation.cpp
  src/impls/ecsim/particles.cpp
)

set(impls_ecsimcorr
  src/impls/ecsimcorr/simulation.cpp
  src/impls/ecsimcorr/particles.cpp
  src/impls/ecsimcorr/energy_conservation.cpp
)

set(impls_ricketson
  src/impls/ricketson/simulation.cpp
  src/impls/ricketson/particles.cpp
)

set(impls_eccapfim
  src/impls/eccapfim/simulation.cpp
  src/impls/eccapfim/particles.cpp
  src/impls/eccapfim/cell_traversal.cpp
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
  ${impls_ecsim}
  ${impls_ecsimcorr}
  ${impls_ricketson}
  ${impls_eccapfim}
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