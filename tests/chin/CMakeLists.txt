cmake_minimum_required(VERSION 3.10)

set(tests
  uniform_b
  linear_b
  curvilinear_b
  crossed_fields
  polarization
  crossed_gradient
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  set_target_properties(${test}.out PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/chin/)
  target_link_libraries(${test}.out ${PROJECT_NAME})
endforeach()
