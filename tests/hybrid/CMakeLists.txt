cmake_minimum_required(VERSION 3.10)

set(comparison_tests
  hybrid_ex1_drift_kinetic
  hybrid_ex1_eccapfim
  hybrid_ex1_ecsimcorr
  hybrid_ex1_hybrid_kinetic
)

foreach(test ${comparison_tests} hybrid_ex2_fms)
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ${MPI_DIR}/bin/mpiexec --bind-to none -n 1 ./${test}.out)
  set_property(TEST "${test}" PROPERTY LABELS impls hybrid)
  set_property(TEST "${test}" PROPERTY RESOURCE_LOCK "hybrid")
endforeach()

# Serialize comparisons so timing.txt measures one run at a time.
foreach(test ${comparison_tests})
  add_test(NAME "${test}_plot" COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/hybrid_ex1_energy.py ${test})
  set_property(TEST "${test}_plot" PROPERTY DEPENDS "${test}")
  set_property(TEST "${test}_plot" PROPERTY LABELS impls hybrid)
  set_property(TEST "${test}_plot" PROPERTY ENVIRONMENT "MPLBACKEND=Agg")
endforeach()

add_test(NAME hybrid_ex2_fms_plot COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/hybrid_ex2_fms.py)
set_property(TEST hybrid_ex2_fms_plot PROPERTY DEPENDS hybrid_ex2_fms)
set_property(TEST hybrid_ex2_fms_plot PROPERTY LABELS impls hybrid)
set_property(TEST hybrid_ex2_fms_plot PROPERTY ENVIRONMENT "MPLBACKEND=Agg")
