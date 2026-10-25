cmake_minimum_required(VERSION 3.10)

set(tests
  ecsim_ex1
  ecsim_ex2
  ecsim_ex3
)

# Setting tolerances manually to stabilize the result
set(ksp_opt
    -ksp_type gmres
    -ksp_atol 1e-14
    -ksp_rtol 1e-14
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ./${test}.out ${ksp_opt})
  set_property(TEST "${test}" PROPERTY LABELS impls ecsim)
  set_property(TEST "${test}" PROPERTY RESOURCE_LOCK "${test}")
  
  add_test(NAME "${test}_mpi_x2" COMMAND mpiexec -np 2 ./${test}.out ${ksp_opt} -da_processors_x 2)
  set_property(TEST "${test}_mpi_x2" PROPERTY LABELS impls ecsim mpi)
  set_property(TEST "${test}_mpi_x2" PROPERTY RESOURCE_LOCK "${test}")
endforeach()

set_property(TEST "ecsim_ex3" PROPERTY DEPENDS "ecsim_ex1;ecsim_ex2")
