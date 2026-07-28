cmake_minimum_required(VERSION 3.10)

set(tests
  #drift_kinetic_push_ex1
  #drift_kinetic_push_ex2
  drift_kinetic_ex1
  drift_kinetic_exb_ex1
  drift_kinetic_gradb_ex2
  drift_kinetic_curv_ex3
  drift_kinetic_curv_gradb_ex4
  drift_kinetic_ex2
  drift_kinetic_energy_ei_ex1
  #drift_kinetic_ex3
  drift_kinetic_mirror_ex1
  #drift_kinetic_sound_ex1
  drift_kinetic_eigen_sound_ex1
  #drift_kinetic_sound_ex3
  #drift_kinetic_betta_ex1
  #drift_kinetic_beta_ex3
  #drift_kinetic_beta_ex5
  #drift_kinetic_hose_ex1
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ${MPI_DIR}/bin/mpiexec --bind-to none -n 1 ./${test}.out)
  set_property(TEST "${test}" PROPERTY LABELS impls eccapfim)
  set_property(TEST "${test}" PROPERTY RUN_SERIAL true)
endforeach()

# Visualization of the ExB drift comparison, executed after the model run.
add_test(
  NAME "drift_kinetic_exb_ex1_plot"
  COMMAND ${CMAKE_SOURCE_DIR}/tests/drift_kinetic/drift_kinetic_exb_ex1.py
)
set_property(TEST "drift_kinetic_exb_ex1_plot" PROPERTY DEPENDS "drift_kinetic_exb_ex1")
set_property(TEST "drift_kinetic_exb_ex1_plot" PROPERTY LABELS impls eccapfim)
# The script only writes a PNG; force the non-interactive Agg backend so it does
# not pull in Qt (and the snap libraries that crash python via LD_LIBRARY_PATH).
set_property(TEST "drift_kinetic_exb_ex1_plot" PROPERTY ENVIRONMENT "MPLBACKEND=Agg")

# Visualization of the grad-B drift comparison, executed after the model run.
add_test(
  NAME "drift_kinetic_gradb_ex2_plot"
  COMMAND ${CMAKE_SOURCE_DIR}/tests/drift_kinetic/drift_kinetic_gradb_ex2.py
)
set_property(TEST "drift_kinetic_gradb_ex2_plot" PROPERTY DEPENDS "drift_kinetic_gradb_ex2")
set_property(TEST "drift_kinetic_gradb_ex2_plot" PROPERTY LABELS impls eccapfim)
set_property(TEST "drift_kinetic_gradb_ex2_plot" PROPERTY ENVIRONMENT "MPLBACKEND=Agg")

# Visualization of the curvature drift comparison, executed after the model run.
add_test(
  NAME "drift_kinetic_curv_ex3_plot"
  COMMAND ${CMAKE_SOURCE_DIR}/tests/drift_kinetic/drift_kinetic_curv_ex3.py
)
set_property(TEST "drift_kinetic_curv_ex3_plot" PROPERTY DEPENDS "drift_kinetic_curv_ex3")
set_property(TEST "drift_kinetic_curv_ex3_plot" PROPERTY LABELS impls eccapfim)
set_property(TEST "drift_kinetic_curv_ex3_plot" PROPERTY ENVIRONMENT "MPLBACKEND=Agg")

# Visualization of the curvature + grad-B drift comparison, executed after the run.
add_test(
  NAME "drift_kinetic_curv_gradb_ex4_plot"
  COMMAND ${CMAKE_SOURCE_DIR}/tests/drift_kinetic/drift_kinetic_curv_gradb_ex4.py
)
set_property(TEST "drift_kinetic_curv_gradb_ex4_plot" PROPERTY DEPENDS "drift_kinetic_curv_gradb_ex4")
set_property(TEST "drift_kinetic_curv_gradb_ex4_plot" PROPERTY LABELS impls eccapfim)
set_property(TEST "drift_kinetic_curv_gradb_ex4_plot" PROPERTY ENVIRONMENT "MPLBACKEND=Agg")

#set(test drift_kinetic_push_ex3)

#add_executable(${test}.out ${test}.cpp)
#target_link_libraries(${test}.out ${PROJECT_NAME})
#
## Python script runs the binary and creates the images of trajectories
#add_test(NAME "${test}" COMMAND ${CMAKE_SOURCE_DIR}/tests/drift_kinetic/${test}_comparison.py)
#set_property(TEST "${test}" PROPERTY LABELS drift_kinetic)
