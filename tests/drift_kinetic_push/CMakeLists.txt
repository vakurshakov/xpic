cmake_minimum_required(VERSION 3.10)

set(tests
  drift_kinetic_push_ex1
  drift_kinetic_push_ex2
  drift_kinetic_push_ex4
  drift_kinetic_push_ex6_1
  drift_kinetic_push_ex6_2
  drift_kinetic_push_ex6_3
  drift_kinetic_push_ex7_1
  drift_kinetic_push_ex7_2
  drift_kinetic_push_ex7_3
  drift_kinetic_push_ex8
)

# In the given tests (Omega * dt) is constant
set(omega_dt 0.1)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}_omega_dt_${omega_dt}" COMMAND ./${test}.out -omega_dt ${omega_dt})
  set_property(TEST "${test}_omega_dt_${omega_dt}" PROPERTY LABELS drift_kinetic)
endforeach()

set(test drift_kinetic_push_ex9)

add_executable(${test}.out ${test}.cpp)
target_link_libraries(${test}.out ${PROJECT_NAME})

# Python script runs the binary and creates the images of trajectories
add_test(NAME "${test}" COMMAND ${CMAKE_SOURCE_DIR}/tests/drift_kinetic_push/${test}_comparison.py)