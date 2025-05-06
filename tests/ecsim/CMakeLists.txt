cmake_minimum_required(VERSION 3.10)

set(tests
  ecsim_ex1
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ./${test}.out)
  add_test(NAME "${test}_mpi_x2" COMMAND mpiexec -np 2 ./${test}.out -da_processors_x 2)

  set_property(TEST "${test}" PROPERTY LABELS impls ecsim)
  set_property(TEST "${test}_mpi_x2" PROPERTY LABELS impls ecsim mpi)
endforeach()
