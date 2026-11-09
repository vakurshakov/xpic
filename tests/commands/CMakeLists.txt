cmake_minimum_required(VERSION 3.10)

set(tests
  dampen_fields
  remove_particles
  set_field
  set_particles
  set_particles_iaw
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ./${test}.out)
  set_property(TEST "${test}" PROPERTY LABELS smoke commands)

  foreach(axis x y z)
    set(name "${test}_mpi_${axis}2")
    add_test(NAME "${name}" COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./${test}.out -da_processors_${axis} 2)
    set_property(TEST "${name}" PROPERTY LABELS smoke commands mpi)
  endforeach()
endforeach()

add_executable(set_particles_halton.out set_particles_halton.cpp)
target_link_libraries(set_particles_halton.out ${PROJECT_NAME})
add_test(NAME set_particles_halton COMMAND ${MPI_DIR}/bin/mpiexec -np 1 ./set_particles_halton.out)
set_property(TEST set_particles_halton PROPERTY LABELS smoke commands)
set_property(TEST set_particles_halton PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
