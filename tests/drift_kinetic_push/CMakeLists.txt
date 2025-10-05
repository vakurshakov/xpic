cmake_minimum_required(VERSION 3.10)

set(tests
  drift_kinetic_push_ex1
  drift_kinetic_push_ex2
  drift_kinetic_push_ex3
  drift_kinetic_push_ex4
  drift_kinetic_push_ex5
  drift_kinetic_grid_boris_ex1
  drift_kinetic_grid_boris_ex2
  drift_kinetic_grid_ex1
  drift_kinetic_grid_ex2
  drift_kinetic_grid_ex3
  drift_kinetic_grid_ex4
)

# In the given tests (Omega * dt) is constant
set(omega_dt 0.1)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(NAME "${test}_omega_dt_${omega_dt}" COMMAND ./${test}.out -omega_dt ${omega_dt})
  set_property(TEST "${test}_omega_dt_${omega_dt}" PROPERTY LABELS drift_kinetic)
endforeach()
