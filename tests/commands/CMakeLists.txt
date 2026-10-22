cmake_minimum_required(VERSION 3.10)

set(tests
  remove_particles
  set_field
  set_particles
  set_particles_iaw
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ./${test}.out)
  set_property(TEST "${test}" PROPERTY LABELS commands)

  add_test(NAME "${test}_mpi_z2" COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./${test}.out -da_processors_z 2)
  set_property(TEST "${test}_mpi_z2" PROPERTY LABELS commands mpi)
endforeach()
