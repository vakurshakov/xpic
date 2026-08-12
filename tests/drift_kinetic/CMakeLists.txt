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
  drift_kinetic_eigen_sound_ex7
  # Archived kinetic-loading experiments retained for reference only:
  #drift_kinetic_eigen_sound_ex10
  #drift_kinetic_eigen_sound_ex11
  #drift_kinetic_eigen_sound_ex12
  #drift_kinetic_eigen_sound_ex13
  #drift_kinetic_eigen_sound_ex14
  drift_kinetic_ringdown_ex0
  drift_kinetic_ringdown_ex1
  drift_kinetic_ringdown_ex2
  drift_kinetic_ringdown_ex3
  drift_kinetic_ringdown_ex4
  drift_kinetic_ringdown_ex5
  drift_kinetic_ringdown_ex6
  drift_kinetic_ringdown_ex7
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

set_property(TEST "drift_kinetic_ringdown_ex0" APPEND PROPERTY
  LABELS free_streaming ringdown)
set_property(TEST "drift_kinetic_ringdown_ex0" PROPERTY
  FIXTURES_SETUP drift_kinetic_free_streaming_run)
set_property(TEST "drift_kinetic_ringdown_ex0" PROPERTY TIMEOUT 3600)

# The density-ringdown cases are intentionally expensive acceptance tests.
# Mark them as such and expose them as setup tests for the quantitative
# post-processing check below.  CTest will therefore include both simulations
# when the analysis test is selected directly.
foreach(test drift_kinetic_ringdown_ex1 drift_kinetic_ringdown_ex2)
  set_property(TEST "${test}" APPEND PROPERTY LABELS ringdown long)
  set_property(TEST "${test}" PROPERTY FIXTURES_SETUP ion_sound_ringdown_runs)
  set_property(TEST "${test}" PROPERTY TIMEOUT 14400)
endforeach()

foreach(test drift_kinetic_ringdown_ex3 drift_kinetic_ringdown_ex4)
  set_property(TEST "${test}" APPEND PROPERTY LABELS ringdown light)
  set_property(TEST "${test}" PROPERTY
    FIXTURES_SETUP ion_sound_ringdown_light_runs)
  set_property(TEST "${test}" PROPERTY TIMEOUT 7200)
endforeach()

# Modern long-wavelength counterpart of the legacy sound_ex19 run.  It is a
# standalone exploratory test rather than a member of the publication/light
# convergence pairs.
set_property(TEST "drift_kinetic_ringdown_ex5" APPEND PROPERTY
  LABELS ringdown long longwave)
set_property(TEST "drift_kinetic_ringdown_ex5" PROPERTY TIMEOUT 43200)

# Faster, better-resolved long-wavelength variant of ex5.
set_property(TEST "drift_kinetic_ringdown_ex6" APPEND PROPERTY
  LABELS ringdown long longwave)
set_property(TEST "drift_kinetic_ringdown_ex6" PROPERTY TIMEOUT 21600)

# Te = 10 keV counterpart of ex1 in a longer box, which halves dt/dz at the
# unchanged dt and brings the electron CFL number below one.  It is a
# standalone test like ex5/ex6, and its marker-step count is several times
# that of ex1, hence the longer timeout.
set_property(TEST "drift_kinetic_ringdown_ex7" APPEND PROPERTY
  LABELS ringdown long)
set_property(TEST "drift_kinetic_ringdown_ex7" PROPERTY TIMEOUT 43200)

add_test(
  NAME "ion_sound_ringdown_synthetic"
  COMMAND ${CMAKE_COMMAND} -E env
    MPLBACKEND=Agg
    MPLCONFIGDIR=/tmp/xpic-matplotlib
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_ion_sound_ringdown.py
)
set_property(TEST "ion_sound_ringdown_synthetic"
  PROPERTY LABELS "drift_kinetic;ringdown;synthetic")

add_test(
  NAME "free_streaming_synthetic"
  COMMAND ${CMAKE_COMMAND} -E env
    MPLBACKEND=Agg
    MPLCONFIGDIR=/tmp/xpic-matplotlib
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_free_streaming.py
)
set_property(TEST "free_streaming_synthetic"
  PROPERTY LABELS "drift_kinetic;free_streaming;synthetic")

add_test(
  NAME "drift_kinetic_free_streaming_analysis"
  COMMAND ${CMAKE_COMMAND} -E env
    MPLBACKEND=Agg
    MPLCONFIGDIR=/tmp/xpic-matplotlib
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/free_streaming.py
    --run ${CMAKE_CURRENT_SOURCE_DIR}/output/drift_kinetic_ringdown_ex0
    --out-dir ${CMAKE_CURRENT_SOURCE_DIR}/output/free_streaming_ex0_analysis
    --check
)
set_property(TEST "drift_kinetic_free_streaming_analysis" PROPERTY
  DEPENDS "drift_kinetic_ringdown_ex0")
set_property(TEST "drift_kinetic_free_streaming_analysis" PROPERTY
  FIXTURES_REQUIRED drift_kinetic_free_streaming_run)
set_property(TEST "drift_kinetic_free_streaming_analysis" PROPERTY
  LABELS "drift_kinetic;free_streaming;analysis")
set_property(TEST "drift_kinetic_free_streaming_analysis" PROPERTY
  RUN_SERIAL true)

add_test(
  NAME "drift_kinetic_ringdown_analysis"
  COMMAND ${CMAKE_COMMAND} -E env
    MPLBACKEND=Agg
    MPLCONFIGDIR=/tmp/xpic-matplotlib
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/ion_sound_ringdown.py
    --runs
      ${CMAKE_CURRENT_SOURCE_DIR}/output/drift_kinetic_ringdown_ex1
      ${CMAKE_CURRENT_SOURCE_DIR}/output/drift_kinetic_ringdown_ex2
    --fit-window 0.5 4.0
    --out-dir ${CMAKE_CURRENT_SOURCE_DIR}/output/ion_sound_ringdown
    --check
)
set_property(TEST "drift_kinetic_ringdown_analysis" PROPERTY
  DEPENDS "drift_kinetic_ringdown_ex1;drift_kinetic_ringdown_ex2")
set_property(TEST "drift_kinetic_ringdown_analysis" PROPERTY
  FIXTURES_REQUIRED ion_sound_ringdown_runs)
set_property(TEST "drift_kinetic_ringdown_analysis" PROPERTY
  LABELS "drift_kinetic;ringdown;long;analysis")
set_property(TEST "drift_kinetic_ringdown_analysis" PROPERTY RUN_SERIAL true)

add_test(
  NAME "drift_kinetic_ringdown_light_analysis"
  COMMAND ${CMAKE_COMMAND} -E env
    MPLBACKEND=Agg
    MPLCONFIGDIR=/tmp/xpic-matplotlib
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/ion_sound_ringdown.py
    --suite light
    --runs
      ${CMAKE_CURRENT_SOURCE_DIR}/output/drift_kinetic_ringdown_ex3
      ${CMAKE_CURRENT_SOURCE_DIR}/output/drift_kinetic_ringdown_ex4
    --fit-window 0.5 2.5
    --out-dir ${CMAKE_CURRENT_SOURCE_DIR}/output/ion_sound_ringdown_light
    --check
)
set_property(TEST "drift_kinetic_ringdown_light_analysis" PROPERTY
  DEPENDS "drift_kinetic_ringdown_ex3;drift_kinetic_ringdown_ex4")
set_property(TEST "drift_kinetic_ringdown_light_analysis" PROPERTY
  FIXTURES_REQUIRED ion_sound_ringdown_light_runs)
set_property(TEST "drift_kinetic_ringdown_light_analysis" PROPERTY
  LABELS "drift_kinetic;ringdown;light;analysis")
set_property(TEST "drift_kinetic_ringdown_light_analysis" PROPERTY
  RUN_SERIAL true)

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
