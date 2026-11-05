cmake_minimum_required(VERSION 3.10)

set(smoke_tests
  drift_kinetic_ex1_segments
  drift_kinetic_ex2_exb
  drift_kinetic_ex3_gradb
  drift_kinetic_ex4_curv
  drift_kinetic_ex6_fields
  drift_kinetic_ex8_periodic_seam
)

set(sanity_tests
  drift_kinetic_ex7_balance
)

set(regression_tests
  drift_kinetic_ex5_iaw
  drift_kinetic_ex10_bounce
  drift_kinetic_ex11_eparallel
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

# Tests also run on a decomposed domain
set(mpi_tests
  drift_kinetic_ex6_fields
  drift_kinetic_ex7_balance
  drift_kinetic_ex8_periodic_seam
)

foreach(level smoke sanity regression)
  foreach(test ${${level}_tests})
    add_executable(${test}.out ${test}.cpp)
    target_link_libraries(${test}.out ${PROJECT_NAME})

    set(opt "")
    if(${test} IN_LIST mpi_tests)
      set(opt ${snes_opt})
    endif()

    add_test(NAME "${test}" COMMAND ${MPI_DIR}/bin/mpiexec -np 1 ./${test}.out ${opt})
    set_property(TEST "${test}" PROPERTY LABELS ${level} impls drift_kinetic)
    set_property(TEST "${test}" PROPERTY RESOURCE_LOCK "drift_kinetic")

    if(${test} IN_LIST mpi_tests)
      foreach(axis x y z)
        add_test(NAME "${test}_mpi_${axis}2" COMMAND mpiexec -np 2 ./${test}.out ${opt} -da_processors_${axis} 2)
        set_property(TEST "${test}_mpi_${axis}2" PROPERTY LABELS ${level} impls drift_kinetic mpi)
        set_property(TEST "${test}_mpi_${axis}2" PROPERTY RESOURCE_LOCK "drift_kinetic")
      endforeach()
    endif()
  endforeach()
endforeach()

set(plot_tests
  drift_kinetic_ex2_exb
  drift_kinetic_ex3_gradb
  drift_kinetic_ex4_curv
  drift_kinetic_ex10_bounce
  drift_kinetic_ex11_eparallel
)

foreach(test ${plot_tests})
  add_test(NAME "${test}_plot" COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/${test}.py)
  set_property(TEST "${test}_plot" PROPERTY DEPENDS "${test}")
  set_property(TEST "${test}_plot" PROPERTY LABELS smoke impls drift_kinetic)
  set_property(TEST "${test}_plot" PROPERTY ENVIRONMENT "MPLBACKEND=Agg")
endforeach()

# One source, two independent field geometries and output directories.
foreach(case aligned oblique)
  set(test drift_kinetic_ex9_nonlin_gradb_${case})
  string(TOUPPER ${case} case_upper)
  add_executable(${test}.out drift_kinetic_ex9_nonlin_gradb.cpp)
  target_compile_definitions(${test}.out PRIVATE NONLIN_GRADB_${case_upper})
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ${MPI_DIR}/bin/mpiexec -np 1 ./${test}.out)
  set_property(TEST "${test}" PROPERTY LABELS regression impls drift_kinetic)
  set_property(TEST "${test}" PROPERTY RESOURCE_LOCK "drift_kinetic")

  add_test(NAME "${test}_plot" COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/drift_kinetic_ex9_nonlin_gradb.py --case ${case})
  set_property(TEST "${test}_plot" PROPERTY DEPENDS "${test}")
  set_property(TEST "${test}_plot" PROPERTY LABELS regression impls drift_kinetic)
  set_property(TEST "${test}_plot" PROPERTY ENVIRONMENT "MPLBACKEND=Agg")
endforeach()

set_property(TEST ${regression_tests} PROPERTY DEPENDS "${smoke_tests};${sanity_tests}")

# Exercise the same seam and continuity checks without cached spline weights.
add_executable(drift_kinetic_seam_raw.out
  drift_kinetic_ex8_periodic_seam.cpp ../../src/algorithms/implicit_drift_kinetic.cpp)
target_compile_definitions(drift_kinetic_seam_raw.out PRIVATE DRIFT_KINETIC_SPLINE_TABLES=0)
target_link_libraries(drift_kinetic_seam_raw.out ${PROJECT_NAME})
add_test(NAME drift_kinetic_seam_raw COMMAND ${MPI_DIR}/bin/mpiexec -np 1 ./drift_kinetic_seam_raw.out)
set_property(TEST drift_kinetic_seam_raw PROPERTY LABELS smoke impls drift_kinetic)
foreach(axis x y z)
  set(test drift_kinetic_seam_raw_mpi_${axis}2)
  add_test(NAME ${test} COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./drift_kinetic_seam_raw.out -da_processors_${axis} 2)
  set_property(TEST ${test} PROPERTY LABELS smoke impls drift_kinetic mpi)
endforeach()
