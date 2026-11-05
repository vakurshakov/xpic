cmake_minimum_required(VERSION 3.10)

set(tests
  drift_kinetic_ex1_segments
  drift_kinetic_ex2_exb
  drift_kinetic_ex3_gradb
  drift_kinetic_ex4_curv
  drift_kinetic_ex5_iaw
  drift_kinetic_ex10_bounce
  drift_kinetic_ex11_eparallel
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ${MPI_DIR}/bin/mpiexec --bind-to none -n 1 ./${test}.out)
  set_property(TEST "${test}" PROPERTY LABELS impls drift_kinetic)
  set_property(TEST "${test}" PROPERTY RESOURCE_LOCK "drift_kinetic")
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
  set_property(TEST "${test}_plot" PROPERTY LABELS impls drift_kinetic)
  set_property(TEST "${test}_plot" PROPERTY ENVIRONMENT "MPLBACKEND=Agg")
endforeach()

# One source, two independent field geometries and output directories.
foreach(case aligned oblique)
  set(test drift_kinetic_ex9_nonlin_gradb_${case})
  string(TOUPPER ${case} case_upper)
  add_executable(${test}.out drift_kinetic_ex9_nonlin_gradb.cpp)
  target_compile_definitions(${test}.out PRIVATE NONLIN_GRADB_${case_upper})
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ${MPI_DIR}/bin/mpiexec --bind-to none -n 1 ./${test}.out)
  set_property(TEST "${test}" PROPERTY LABELS impls drift_kinetic)
  set_property(TEST "${test}" PROPERTY RESOURCE_LOCK "drift_kinetic")

  add_test(NAME "${test}_plot" COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/drift_kinetic_ex9_nonlin_gradb.py --case ${case})
  set_property(TEST "${test}_plot" PROPERTY DEPENDS "${test}")
  set_property(TEST "${test}_plot" PROPERTY LABELS impls drift_kinetic)
  set_property(TEST "${test}_plot" PROPERTY ENVIRONMENT "MPLBACKEND=Agg")
endforeach()
