cmake_minimum_required(VERSION 3.10)

set(tests
  ecsimcorr_ex1
  ecsimcorr_ex2
  ecsimcorr_ex3
  ecsimcorr_ex4
)

# Setting tolerances manually to stabilize the result
set(ksp_opt
    -predict_ksp_type gmres
    -predict_ksp_atol 1e-08
    -predict_ksp_rtol 1e-08
    -correct_ksp_type gmres
    -correct_ksp_atol 1e-14
    -correct_ksp_rtol 1e-14
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ./${test}.out ${ksp_opt})
  set_property(TEST "${test}" PROPERTY LABELS impls ecsimcorr)
  set_property(TEST "${test}" PROPERTY RESOURCE_LOCK "${test}")

  add_test(NAME "${test}_mpi_x2" COMMAND mpiexec -np 2 ./${test}.out ${ksp_opt} -da_processors_x 2)
  set_property(TEST "${test}_mpi_x2" PROPERTY LABELS impls ecsimcorr mpi)
  set_property(TEST "${test}_mpi_x2" PROPERTY RESOURCE_LOCK "${test}")
endforeach()

set_property(TEST "ecsimcorr_ex4" PROPERTY DEPENDS "ecsimcorr_ex1;ecsimcorr_ex2;ecsimcorr_ex3")
