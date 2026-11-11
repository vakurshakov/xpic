cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# clangd (see .clangd) reads the database of the last configured build type.
file(CREATE_LINK ${CMAKE_BINARY_DIR}/compile_commands.json
  ${CMAKE_SOURCE_DIR}/build/compile_commands.json SYMBOLIC)

# Absolute paths, because they end up in rpaths.
get_filename_component(MPI_DIR "$ENV{MPI_DIR}" ABSOLUTE)
get_filename_component(JSON_DIR "$ENV{JSON_DIR}" ABSOLUTE)
get_filename_component(PETSC_DIR "$ENV{PETSC_DIR}" ABSOLUTE)

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
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -mno-wbnoinvd")

# PETSC_ARCH from the environment overrides the per-configuration defaults.
if(DEFINED ENV{PETSC_ARCH})
  set(PETSC_ARCH_OPT $ENV{PETSC_ARCH})
  set(PETSC_ARCH_DEBUG $ENV{PETSC_ARCH})
else()
  set(PETSC_ARCH_OPT linux-mpi-opt)
  set(PETSC_ARCH_DEBUG linux-mpi-debug)
endif()
set(PETSC_LIB_DIR ${PETSC_DIR}/$<IF:$<CONFIG:Release>,${PETSC_ARCH_OPT},${PETSC_ARCH_DEBUG}>/lib)

# Clang needs the libstdc++ of a GCC installation, which GCC_DIR points to.
set(GCC_DIR $ENV{GCC_DIR})
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND GCC_DIR)
  file(GLOB gcc_cxx_dirs ${GCC_DIR}/include/c++/*)
  list(SORT gcc_cxx_dirs COMPARE NATURAL)
  list(POP_BACK gcc_cxx_dirs gcc_cxx_dir)
  file(GLOB gcc_cxx_target_dirs LIST_DIRECTORIES true ${gcc_cxx_dir}/*-*)
  string(APPEND CMAKE_CXX_FLAGS " --gcc-toolchain=${GCC_DIR} -isystem ${gcc_cxx_dir}")
  foreach(dir ${gcc_cxx_target_dirs})
    string(APPEND CMAKE_CXX_FLAGS " -isystem ${dir}")
  endforeach()
  set(GCC_RPATH "-Wl,-rpath,${GCC_DIR}/lib64")
endif()

set(interfaces
  src/interfaces/simulation.cpp
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
  src/diagnostics/mat_mult_field_view.cpp
  src/diagnostics/energy.cpp
  src/diagnostics/distribution_function.cpp
  src/diagnostics/distribution_moment.cpp
  src/diagnostics/mat_dump.cpp
  src/diagnostics/log_view.cpp
  src/diagnostics/simulation_backup.cpp
  src/diagnostics/charge_conservation.cpp
  src/diagnostics/momentum_conservation.cpp
  src/diagnostics/table_diagnostic.cpp
  src/diagnostics/particle_trace.cpp
  src/diagnostics/builders/diagnostic_builder.cpp
  src/diagnostics/builders/field_view_builder.cpp
  src/diagnostics/builders/field_view_zavg_builder.cpp
  src/diagnostics/builders/mat_mult_field_view_builder.cpp
  src/diagnostics/builders/distribution_function_builder.cpp
  src/diagnostics/builders/distribution_moment_builder.cpp
  src/diagnostics/builders/log_view_builder.cpp
  src/diagnostics/builders/momentum_conservation_builder.cpp
  src/diagnostics/builders/particle_trace_builder.cpp
  src/diagnostics/builders/simulation_backup_builder.cpp
)

set(commands
  src/commands/inject_particles.cpp
  src/commands/set_particles.cpp
  src/commands/set_particles_iaw.cpp
  src/commands/set_field.cpp
  src/commands/remove_particles.cpp
  src/commands/dampen_fields.cpp
  src/commands/builders/command_builder.cpp
  src/commands/builders/particles_builder.cpp
  src/commands/builders/inject_particles_builder.cpp
  src/commands/builders/remove_particles_builder.cpp
  src/commands/builders/set_particles_builder.cpp
  src/commands/builders/set_particles_iaw_builder.cpp
  src/commands/builders/set_field_builder.cpp
  src/commands/builders/dampen_fields_builder.cpp
)

set(utils
  src/utils/utils.cpp
  src/utils/configuration.cpp
  src/utils/profiler.cpp
  src/utils/operators.cpp
  src/utils/shape.cpp
  src/utils/particle_storage.cpp
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
  PUBLIC ${PETSC_LIB_DIR}
)

target_link_options(
  ${PROJECT_NAME}
  PUBLIC -Wl,-rpath,${MPI_DIR}/lib
  PUBLIC -Wl,-rpath,${PETSC_LIB_DIR}
  PUBLIC ${GCC_RPATH}
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
