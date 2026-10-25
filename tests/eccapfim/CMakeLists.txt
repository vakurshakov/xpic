cmake_minimum_required(VERSION 3.10)

set(tests
  eccapfim_ex1
  eccapfim_ex2
  eccapfim_ex3
  eccapfim_ex4
)

# Setting snes options manually to stabilize the result
set(snes_opt
    -snes_type anderson
    -snes_anderson_m 40
    -snes_anderson_restart 40
    -snes_anderson_beta 0.65

    -snes_atol 1e-10
    -snes_rtol 1e-10
    -snes_stol 1e-6
    -snes_max_it 35

    -snes_ksp_ew
    -snes_ksp_ew_version 3
    -snes_ksp_ew_rtol0 0.1
    -snes_ksp_ew_rtolmax 0.9
    -snes_ksp_ew_gamma 0.9
    -snes_ksp_ew_alpha 1.5
    -snes_ksp_ew_alpha2 1.5
    -snes_ksp_ew_threshold 0.1
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ./${test}.out ${snes_opt})
  set_property(TEST "${test}" PROPERTY LABELS impls eccapfim)
  set_property(TEST "${test}" PROPERTY RESOURCE_LOCK "${test}")

  add_test(NAME "${test}_mpi_x2" COMMAND mpiexec -np 2 ./${test}.out ${snes_opt} -da_processors_x 2)
  set_property(TEST "${test}_mpi_x2" PROPERTY LABELS impls eccapfim mpi)
  set_property(TEST "${test}_mpi_x2" PROPERTY RESOURCE_LOCK "${test}")
endforeach()

set_property(TEST "eccapfim_ex4" PROPERTY DEPENDS "eccapfim_ex1;eccapfim_ex2;eccapfim_ex3")
