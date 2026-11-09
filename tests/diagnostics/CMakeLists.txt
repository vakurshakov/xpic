cmake_minimum_required(VERSION 3.10)

set(tests
  field_view
  field_view_zavg
  mat_mult_field_view
  distribution_function
  distribution_function_dk
  distribution_moment
  distribution_moment_dk
  distribution_moment_deposit
  simulation_backup
  simulation_backup_dk
)

# Variants of one test share output files, so they hold a common lock and run one at a time.
foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ./${test}.out)
  set_property(TEST "${test}" PROPERTY LABELS smoke diagnostics)
  set_property(TEST "${test}" PROPERTY RESOURCE_LOCK "${test}")

  foreach(axis x y z)
    # @todo Fix `FieldViewZAvg` for runs decomposed along z
    if(${test} STREQUAL "field_view_zavg" AND ${axis} STREQUAL "z")
        continue()
    endif()

    set(name "${test}_mpi_${axis}2")
    add_test(NAME "${name}" COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./${test}.out -da_processors_${axis} 2)
    set_property(TEST "${name}" PROPERTY LABELS smoke diagnostics mpi)
    set_property(TEST "${name}" PROPERTY RESOURCE_LOCK "${test}")
  endforeach()
endforeach()

# Restart from a backup must reproduce the continuous run, for kinetic and drift-kinetic sorts
add_executable(simulation_restart.out simulation_restart.cpp)
target_link_libraries(simulation_restart.out ${PROJECT_NAME})

foreach(simulation eccapfim drift_kinetic)
  set(test "simulation_restart_${simulation}")
  add_test(NAME "${test}" COMMAND ./simulation_restart.out -simulation ${simulation})
  set_property(TEST "${test}" PROPERTY LABELS sanity diagnostics)
  set_property(TEST "${test}" PROPERTY RESOURCE_LOCK "simulation_restart")

  foreach(axis x y z)
    set(name "${test}_mpi_${axis}2")
    add_test(NAME "${name}" COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./simulation_restart.out -simulation ${simulation} -da_processors_${axis} 2)
    set_property(TEST "${name}" PROPERTY LABELS sanity diagnostics mpi)
    set_property(TEST "${name}" PROPERTY RESOURCE_LOCK "simulation_restart")
  endforeach()
endforeach()

# One file set for all schemes, so they hold a common lock and run one at a time.
add_executable(log_view.out log_view.cpp)
target_link_libraries(log_view.out ${PROJECT_NAME})

foreach(simulation ecsim ecsimcorr eccapfim)
  set(test "log_view_${simulation}")

  add_test(NAME "${test}" COMMAND ./log_view.out -simulation ${simulation})
  set_property(TEST "${test}" PROPERTY LABELS smoke diagnostics ${simulation})
  set_property(TEST "${test}" PROPERTY RESOURCE_LOCK "log_view")

  add_test(NAME "${test}_mpi_x2" COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./log_view.out -simulation ${simulation} -da_processors_x 2)
  set_property(TEST "${test}_mpi_x2" PROPERTY LABELS smoke diagnostics ${simulation} mpi)
  set_property(TEST "${test}_mpi_x2" PROPERTY RESOURCE_LOCK "log_view")
endforeach()
