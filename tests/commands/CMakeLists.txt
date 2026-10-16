cmake_minimum_required(VERSION 3.10)

set(tests
  set_field
  set_particles
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ./${test}.out)
  set_property(TEST "${test}" PROPERTY LABELS commands)
endforeach()

add_test(NAME "set_field_mpi_x1y1z2" COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./set_field.out -da_processors_z 2)
set_property(TEST "set_field_mpi_x1y1z2" PROPERTY LABELS commands mpi)
set_property(TEST "set_field_mpi_x1y1z2" PROPERTY RUN_SERIAL true)

add_test(NAME "set_particles_mpi_x1y1z2" COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./set_particles.out -da_processors_z 2)
set_property(TEST "set_particles_mpi_x1y1z2" PROPERTY LABELS commands mpi)
set_property(TEST "set_particles_mpi_x1y1z2" PROPERTY RUN_SERIAL true)
