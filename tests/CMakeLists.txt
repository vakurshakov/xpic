cmake_minimum_required(VERSION 3.29)

add_executable(gyration.out gyration.cpp)
target_link_libraries(gyration.out ${PROJECT_NAME})
