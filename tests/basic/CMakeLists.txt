cmake_minimum_required(VERSION 3.10)

set(tests
  basic_ex1
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ./${test}.out)
  set_property(TEST "${test}" PROPERTY LABELS impls basic)
  set_property(TEST "${test}" PROPERTY RUN_SERIAL true)

  add_test(NAME "${test}_mpi_x2" COMMAND mpiexec -np 2 ./${test}.out -da_processors_x 2)
  set_property(TEST "${test}_mpi_x2" PROPERTY LABELS impls basic mpi)
  set_property(TEST "${test}_mpi_x2" PROPERTY RUN_SERIAL true)
endforeach()

add_executable(particles_load_unit.out particles_load_unit.cpp)
target_link_libraries(particles_load_unit.out ${PROJECT_NAME})
add_test(NAME particles_load_unit COMMAND ./particles_load_unit.out)
set_property(TEST particles_load_unit PROPERTY LABELS utils particles_load)

add_executable(drift_kinetic_segments_unit.out drift_kinetic_segments_unit.cpp)
target_link_libraries(drift_kinetic_segments_unit.out ${PROJECT_NAME})
add_test(NAME drift_kinetic_segments_unit COMMAND ./drift_kinetic_segments_unit.out)
set_property(TEST drift_kinetic_segments_unit PROPERTY
  LABELS utils drift_kinetic segments)

add_executable(drift_kinetic_adaptive_push_unit.out
  drift_kinetic_adaptive_push_unit.cpp)
target_link_libraries(drift_kinetic_adaptive_push_unit.out ${PROJECT_NAME})
add_test(NAME drift_kinetic_adaptive_push_unit
  COMMAND ./drift_kinetic_adaptive_push_unit.out)
set_property(TEST drift_kinetic_adaptive_push_unit PROPERTY
  LABELS utils drift_kinetic pusher)
