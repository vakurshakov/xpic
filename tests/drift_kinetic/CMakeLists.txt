cmake_minimum_required(VERSION 3.10)

set(tests
  drift_kinetic_exb_ex1
  drift_kinetic_gradb_ex2
  drift_kinetic_curv_ex3
  drift_kinetic_ion_sound_ex5
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(
    NAME "${test}"
    COMMAND ${MPI_DIR}/bin/mpiexec --bind-to none -n 1 ./${test}.out
  )
  set_property(TEST "${test}" PROPERTY LABELS impls drift_kinetic)
  set_property(TEST "${test}" PROPERTY RUN_SERIAL true)
endforeach()

set(plot_tests
  drift_kinetic_exb_ex1
  drift_kinetic_gradb_ex2
  drift_kinetic_curv_ex3
)

foreach(test ${plot_tests})
  add_test(
    NAME "${test}_plot"
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/${test}.py
  )
  set_property(TEST "${test}_plot" PROPERTY DEPENDS "${test}")
  set_property(TEST "${test}_plot" PROPERTY LABELS impls drift_kinetic)
  set_property(TEST "${test}_plot" PROPERTY ENVIRONMENT "MPLBACKEND=Agg")
endforeach()
