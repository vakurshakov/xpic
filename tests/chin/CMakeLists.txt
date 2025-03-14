cmake_minimum_required(VERSION 3.10)

set(tests
  uniform_b
)

# linear_b
# curvilinear_b
# crossed_fields
# polarization
# crossed_gradient

set(uniform_b_ids
  M1A
  M1B

  B1A
  B1B

  C1A
  C1B

  MLF
  BLF
  CLF

  M2A
  M2B
  C2A
  B2B
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  foreach(id ${${test}_ids})
    add_test(
      NAME "${test}_${id}"
      COMMAND ${test}.out -id ${id}
      PROPERTY LABELS chin
    )
  endforeach()
endforeach()
