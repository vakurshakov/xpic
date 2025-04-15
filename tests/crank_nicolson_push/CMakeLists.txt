cmake_minimum_required(VERSION 3.10)

set(tests
  crank_nicolson_push_ex1
)

set(omega_dt_values
  0.1
  1.0
  10.0
  100.0
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  foreach(omega_dt ${omega_dt_values})
    add_test(NAME "${test}_OmegaDt${omega_dt}" COMMAND ./${test}.out -omega_dt ${omega_dt})
    set_property(TEST "${test}_OmegaDt${omega_dt}" PROPERTY LABELS crank_nicolson)
  endforeach()
endforeach()
