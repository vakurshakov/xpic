cmake_minimum_required(VERSION 3.10)

# Benchmarks, not tests, run by hand

add_executable(particle_storage_suite.out particle_storage/suite.cpp)
target_link_libraries(particle_storage_suite.out ${PROJECT_NAME})
