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

# Reproduces the `DistributionMoment` defect (region-gated deposit with its own 1st-order shape).
# Drop WILL_FAIL when the full-world deposit lands.
foreach(name distribution_moment_deposit distribution_moment_deposit_mpi_x2 distribution_moment_deposit_mpi_y2 distribution_moment_deposit_mpi_z2)
  set_property(TEST ${name} PROPERTY WILL_FAIL TRUE)
endforeach()

# Restart from a backup must reproduce the continuous run, for kinetic and drift-kinetic sorts
add_executable(simulation_restart.out simulation_restart.cpp)
target_link_libraries(simulation_restart.out ${PROJECT_NAME})

foreach(simulation eccapfim drift_kinetic)
  set(test "simulation_restart_${simulation}")
  add_test(NAME "${test}" COMMAND ./simulation_restart.out -restart_simulation ${simulation})
  set_property(TEST "${test}" PROPERTY LABELS sanity diagnostics)
  set_property(TEST "${test}" PROPERTY RESOURCE_LOCK "simulation_restart")

  foreach(axis x y z)
    set(name "${test}_mpi_${axis}2")
    add_test(NAME "${name}" COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./simulation_restart.out -restart_simulation ${simulation} -da_processors_${axis} 2)
    set_property(TEST "${name}" PROPERTY LABELS sanity diagnostics mpi)
    set_property(TEST "${name}" PROPERTY RESOURCE_LOCK "simulation_restart")
  endforeach()
endforeach()

# Known defect of the DK pusher on multi-rank runs (see `drift_kinetic_ex7_balance_mpi_*`): the
# continuous run itself diverges here, before any restart. Drop WILL_FAIL when it is fixed.
set_property(TEST simulation_restart_drift_kinetic_mpi_x2 PROPERTY WILL_FAIL TRUE)
