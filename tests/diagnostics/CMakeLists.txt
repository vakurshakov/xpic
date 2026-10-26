cmake_minimum_required(VERSION 3.10)

set(tests
  field_view
  field_view_zavg
  mat_mult_field_view
  distribution_function
  distribution_function_dk
  distribution_moment
  distribution_moment_dk
  simulation_backup
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
