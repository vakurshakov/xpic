cmake_minimum_required(VERSION 3.10)

set(smoke_tests
  basic_ex1
)

set(sanity_tests
  basic_ex2
  basic_ex3
)

set(regression_tests
  basic_ex4
)

foreach(level smoke sanity regression)
  foreach(test ${${level}_tests})
    add_executable(${test}.out ${test}.cpp)
    target_link_libraries(${test}.out ${PROJECT_NAME})

    add_test(NAME "${test}" COMMAND ./${test}.out)
    set_property(TEST "${test}" PROPERTY LABELS ${level} impls basic)
    set_property(TEST "${test}" PROPERTY RESOURCE_LOCK "${test}")

    add_test(NAME "${test}_mpi_x2" COMMAND mpiexec -np 2 ./${test}.out -da_processors_x 2)
    set_property(TEST "${test}_mpi_x2" PROPERTY LABELS ${level} impls basic mpi)
    set_property(TEST "${test}_mpi_x2" PROPERTY RESOURCE_LOCK "${test}")
  endforeach()
endforeach()

set_property(TEST ${sanity_tests} PROPERTY DEPENDS "${smoke_tests}")
set_property(TEST ${regression_tests} PROPERTY DEPENDS "${smoke_tests};${sanity_tests}")
