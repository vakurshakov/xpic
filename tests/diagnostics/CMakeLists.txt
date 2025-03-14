cmake_minimum_required(VERSION 3.10)

set(tests
  field_view
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  add_test(
    NAME "${test}"
    COMMAND ${test}.out
  )

  set_property(
    TEST "${test}"
    PROPERTY LABELS diagnostics
  )
endforeach()
