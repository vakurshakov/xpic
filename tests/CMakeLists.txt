cmake_minimum_required(VERSION 3.29)

# @todo check ctest
# @todo replace with google-tests environment
set(tests
  chin_uniform_b
  chin_linear_b
  chin_curvilinear_b
  chin_crossed_fields
  chin_polarization
  gyration
  drift_crossed_fields
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})
endforeach()
