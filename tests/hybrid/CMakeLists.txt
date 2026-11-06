cmake_minimum_required(VERSION 3.10)

add_executable(hybrid_storage.out hybrid_storage.cpp)
target_link_libraries(hybrid_storage.out ${PROJECT_NAME})

foreach(mode evolving frozen preconditioned)
  set(opt "")
  if(mode STREQUAL "frozen")
    set(opt -hybrid_frozen)
  elseif(mode STREQUAL "preconditioned")
    set(opt -hybrid_preconditioner)
  endif()
  set(test hybrid_storage_${mode})
  add_test(NAME ${test} COMMAND ${MPI_DIR}/bin/mpiexec -np 1 ./hybrid_storage.out ${opt})
  set_property(TEST ${test} PROPERTY LABELS smoke impls hybrid)
  set_property(TEST ${test} PROPERTY RESOURCE_LOCK hybrid_storage)
  set_property(TEST ${test} PROPERTY TIMEOUT 30)
  foreach(axis x y z)
    set(test hybrid_storage_${mode}_mpi_${axis}2)
    add_test(NAME ${test} COMMAND ${MPI_DIR}/bin/mpiexec -np 2 ./hybrid_storage.out ${opt} -da_processors_${axis} 2)
    set_property(TEST ${test} PROPERTY LABELS smoke impls hybrid mpi)
    set_property(TEST ${test} PROPERTY RESOURCE_LOCK hybrid_storage)
    set_property(TEST ${test} PROPERTY TIMEOUT 30)
  endforeach()
endforeach()

set(comparison_tests
  hybrid_ex1_drift_kinetic
  hybrid_ex1_eccapfim
  hybrid_ex1_ecsimcorr
  hybrid_ex1_hybrid_kinetic
)

add_executable(hybrid_ex1.out hybrid_ex1.cpp)
target_link_libraries(hybrid_ex1.out ${PROJECT_NAME})

foreach(test ${comparison_tests})
  string(REPLACE "hybrid_ex1_" "" model ${test})
  add_test(NAME "${test}" COMMAND ${MPI_DIR}/bin/mpiexec -np 1 ./hybrid_ex1.out -ex1_model ${model})
  set_property(TEST "${test}" PROPERTY LABELS regression impls hybrid)
  set_property(TEST "${test}" PROPERTY RESOURCE_LOCK "hybrid")
endforeach()

add_executable(hybrid_ex2_fms.out hybrid_ex2_fms.cpp)
target_link_libraries(hybrid_ex2_fms.out ${PROJECT_NAME})
add_test(NAME hybrid_ex2_fms COMMAND ${MPI_DIR}/bin/mpiexec -np 1 ./hybrid_ex2_fms.out)
set_property(TEST hybrid_ex2_fms PROPERTY LABELS regression impls hybrid)
set_property(TEST hybrid_ex2_fms PROPERTY RESOURCE_LOCK "hybrid")

# Serialize comparisons so timing.txt measures one run at a time.
foreach(test ${comparison_tests})
  add_test(NAME "${test}_plot" COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/hybrid_ex1_energy.py ${test})
  set_property(TEST "${test}_plot" PROPERTY DEPENDS "${test}")
  set_property(TEST "${test}_plot" PROPERTY LABELS regression impls hybrid)
  set_property(TEST "${test}_plot" PROPERTY ENVIRONMENT "MPLBACKEND=Agg")
endforeach()

add_test(NAME hybrid_ex2_fms_plot COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/hybrid_ex2_fms.py)
set_property(TEST hybrid_ex2_fms_plot PROPERTY DEPENDS hybrid_ex2_fms)
set_property(TEST hybrid_ex2_fms_plot PROPERTY LABELS regression impls hybrid)
set_property(TEST hybrid_ex2_fms_plot PROPERTY ENVIRONMENT "MPLBACKEND=Agg")

add_test(NAME hybrid_storage_evolving_mpi_x3
  COMMAND ${MPI_DIR}/bin/mpiexec -np 3 ./hybrid_storage.out -da_processors_x 3)
set_property(TEST hybrid_storage_evolving_mpi_x3 PROPERTY LABELS smoke impls hybrid mpi)
set_property(TEST hybrid_storage_evolving_mpi_x3 PROPERTY RESOURCE_LOCK hybrid_storage)
set_property(TEST hybrid_storage_evolving_mpi_x3 PROPERTY TIMEOUT 30)
