cmake_minimum_required(VERSION 3.10)

set(tests
  drift_kinetic_ex1_segments
  drift_kinetic_ex2_exb
  drift_kinetic_ex3_gradb
  drift_kinetic_ex4_curv
  drift_kinetic_ex5_iaw
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
)

foreach(test ${plot_tests})
  add_test(NAME "${test}_plot" COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/${test}.py)
  set_property(TEST "${test}_plot" PROPERTY DEPENDS "${test}")
  set_property(TEST "${test}_plot" PROPERTY LABELS impls drift_kinetic)
  set_property(TEST "${test}_plot" PROPERTY ENVIRONMENT "MPLBACKEND=Agg")
endforeach()
