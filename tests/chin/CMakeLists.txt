cmake_minimum_required(VERSION 3.10)

set(tests
  uniform_b
  linear_b
  curvilinear_b
  crossed_fields
  polarization
  crossed_gradient
)

set(uniform_b_ids
  M1A M1B
  B1A B1B
  C1A C1B
  MLF BLF CLF
  M2A M2B C2A B2B
)

set(B_IDS B1A B1B B2B BLF)
set(EB_IDS EB1A EB1B EB2B EBLF)

set(linear_b_ids ${B_IDS})
set(curvilinear_b_ids ${B_IDS})

set(crossed_fields_ids ${EB_IDS})
set(polarization_ids ${EB_IDS})
set(crossed_gradient_ids ${EB_IDS})

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  foreach(id ${${test}_ids})
    add_test(
      NAME "${test}_${id}"
      COMMAND ${test}.out -id ${id}
    )

    set_property(
      TEST "${test}_${id}"
      PROPERTY LABELS chin
    )
  endforeach()
endforeach()
