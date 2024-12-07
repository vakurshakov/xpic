cmake_minimum_required(VERSION 3.29)

# @todo check ctest
# @todo replace with google-tests environment
set(tests
  chin_gyration
  gyration
  drift_crossed_fields
)

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})
endforeach()
