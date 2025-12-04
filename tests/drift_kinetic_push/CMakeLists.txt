cmake_minimum_required(VERSION 3.10)

set(tests
  drift_kinetic_push_ex1
  drift_kinetic_push_ex2
  drift_kinetic_push_ex3
  drift_kinetic_push_ex4
  drift_kinetic_push_ex5
  drift_kinetic_interpolation_ex1
  drift_kinetic_interpolation_ex2
  drift_kinetic_interpolation_ex3
  drift_kinetic_grid_boris_ex1
  drift_kinetic_grid_boris_ex2
  drift_kinetic_grid_boris_ex3
  drift_kinetic_grid_boris_ex4
)

# In the given tests (Omega * dt) is constant
set(omega_dt 0.1)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}" COMMAND ./${test}.out -omega_dt ${omega_dt})
  set_property(TEST "${test}" PROPERTY LABELS drift_kinetic)
endforeach()

set(test drift_kinetic_push_ex9)

add_executable(${test}.out ${test}.cpp)
target_link_libraries(${test}.out ${PROJECT_NAME})

# Python script runs the binary and creates the images of trajectories
add_test(NAME "${test}" COMMAND ${CMAKE_SOURCE_DIR}/tests/drift_kinetic_push/${test}_comparison.py)