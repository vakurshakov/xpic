cmake_minimum_required(VERSION 3.10)

set(tests
  field_view
  simulation_backup
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ./${test}.out)
  set_property(TEST "${test}" PROPERTY LABELS diagnostics)
endforeach()

add_test(NAME "simulation_backup_mpi_x1y1z2" COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./simulation_backup.out -da_processors_z 2)
set_property(TEST "simulation_backup_mpi_x1y1z2" PROPERTY LABELS diagnostics mpi)

add_test(NAME "simulation_backup_mpi_x1y2z1" COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./simulation_backup.out -da_processors_y 2)
set_property(TEST "simulation_backup_mpi_x1y2z1" PROPERTY LABELS diagnostics mpi)

add_test(NAME "simulation_backup_mpi_x2y1z1" COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./simulation_backup.out -da_processors_x 2)
set_property(TEST "simulation_backup_mpi_x2y1z1" PROPERTY LABELS diagnostics mpi)