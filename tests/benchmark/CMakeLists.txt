cmake_minimum_required(VERSION 3.10)

# Benchmark, not a test: run by hand, see storage_spike.cpp.
add_executable(storage_spike.out storage_spike.cpp)
target_link_libraries(storage_spike.out ${PROJECT_NAME})
