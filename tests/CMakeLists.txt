cmake_minimum_required(VERSION 3.10)

add_subdirectory(boris_push)
add_subdirectory(crank_nicolson_push)

add_subdirectory(commands)
add_subdirectory(diagnostics)
add_subdirectory(utils)

add_subdirectory(basic)
add_subdirectory(eccapfim)
add_subdirectory(ecsim)
add_subdirectory(ecsimcorr)
add_subdirectory(drift_kinetic)
