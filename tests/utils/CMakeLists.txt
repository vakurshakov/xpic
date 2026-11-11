cmake_minimum_required(VERSION 3.10)

set(tests
  operators
  particle_storage
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ./${test}.out)
  set_property(TEST "${test}" PROPERTY LABELS smoke utils)

  foreach(axis x y z)
    set(name "${test}_mpi_${axis}2")
    add_test(NAME "${name}" COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./${test}.out -da_processors_${axis} 2)
    set_property(TEST "${name}" PROPERTY LABELS smoke utils mpi)
  endforeach()
endforeach()

add_executable(coordinate_generators_qs.out coordinate_generators_qs.cpp)
target_link_libraries(coordinate_generators_qs.out ${PROJECT_NAME})
add_test(NAME coordinate_generators_qs COMMAND ./coordinate_generators_qs.out)
set_property(TEST coordinate_generators_qs PROPERTY LABELS smoke utils)

# Edge and corner neighbours. The domain is 16x18x16 cells (y splits unevenly over 4 ranks)
# and the DMDA needs at least 4 cells per rank, so x4 is the finest split along x.
add_test(
  NAME particle_storage_mpi_x2y2z2
  COMMAND ${MPI_DIR}/bin/mpiexec
    -np 8 ./particle_storage.out
    -da_processors_x 2
    -da_processors_y 2
    -da_processors_z 2
)
add_test(
  NAME particle_storage_mpi_x4y4z1
  COMMAND ${MPI_DIR}/bin/mpiexec
    --oversubscribe -np 16 ./particle_storage.out
    -da_processors_x 4
    -da_processors_y 4
    -da_processors_z 1
)
foreach(test particle_storage_mpi_x2y2z2 particle_storage_mpi_x4y4z1)
  set_property(TEST ${test} PROPERTY LABELS smoke utils mpi)
endforeach()

add_test(NAME particle_storage_mpi_x3
  COMMAND ${MPI_DIR}/bin/mpiexec -np 3 ./particle_storage.out -da_processors_x 3)
set_property(TEST particle_storage_mpi_x3 PROPERTY LABELS smoke utils mpi)
