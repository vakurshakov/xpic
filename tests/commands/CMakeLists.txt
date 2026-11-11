cmake_minimum_required(VERSION 3.10)

set(tests
  dampen_fields
  remove_particles
  set_field
  set_particles
  set_particles_iaw
  set_particles_qs
)

foreach(test ${tests})
  set(test_labels smoke commands)
  if(test STREQUAL "set_particles_qs")
    set(test_labels sanity commands drift_kinetic)
  endif()

  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ./${test}.out)
  set_property(TEST "${test}" PROPERTY LABELS ${test_labels})

  foreach(axis x y z)
    set(name "${test}_mpi_${axis}2")
    add_test(NAME "${name}" COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./${test}.out -da_processors_${axis} 2)
    set_property(TEST "${name}" PROPERTY LABELS ${test_labels} mpi)
  endforeach()
endforeach()
