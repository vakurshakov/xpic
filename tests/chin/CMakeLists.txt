cmake_minimum_required(VERSION 3.10)

set(tests
  boris_push_ex1
  boris_push_ex2
  boris_push_ex3
  boris_push_ex4
  boris_push_ex5
  boris_push_ex6
)

set(boris_push_ex1_ids
  M1A M1B
  B1A B1B
  C1A C1B
  MLF BLF CLF
  M2A M2B
  C2A B2B
)

set(B_IDS B1A B1B B2B BLF)
set(EB_IDS EB1A EB1B EB2B EBLF)

set(boris_push_ex2_ids ${B_IDS})
set(boris_push_ex3_ids ${B_IDS})

set(boris_push_ex4_ids ${EB_IDS})
set(boris_push_ex5_ids ${EB_IDS})
set(boris_push_ex6_ids ${EB_IDS})

foreach(test ${tests})
  add_executable(${test}.out ${test}.cpp)
  target_link_libraries(${test}.out ${PROJECT_NAME})

  foreach(id ${${test}_ids})
    add_test(NAME "${test}_${id}" COMMAND ./${test}.out -id ${id})
    set_property(TEST "${test}_${id}" PROPERTY LABELS chin boris)
  endforeach()
endforeach()
