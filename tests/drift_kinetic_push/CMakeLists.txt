cmake_minimum_required(VERSION 3.10)

set(tests
  drift_kinetic_push_ex1
  drift_kinetic_push_ex2
  drift_kinetic_push_ex3
  drift_kinetic_push_ex4
  drift_kinetic_push_ex5
  drift_kinetic_push_ex6_1
  drift_kinetic_push_ex6_2
  drift_kinetic_push_ex6_3
  drift_kinetic_push_ex7_1
  drift_kinetic_push_ex7_2
  drift_kinetic_push_ex7_3
  drift_kinetic_push_ex8
  )

# Since Crank-Nicolson particle integrator enables large timesteps to 
# be taken, we will _test_ against exactly this property of the mover.
set(omega_dt_values
  0.1
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  foreach(omega_dt ${omega_dt_values})
    add_test(NAME "${test}_omega_dt_${omega_dt}" COMMAND ./${test}.out -omega_dt ${omega_dt})
    set_property(TEST "${test}_omega_dt_${omega_dt}" PROPERTY LABELS drift_kinetic)
  endforeach()
endforeach()
