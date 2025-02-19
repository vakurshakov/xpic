#!/bin/bash

source ./header.sh

usage() { echo "Usage: $0 <config.json>" 1>&2; exit 1; }

export MPI_NUM_PROC=2

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=1

# To inspect thread migration
# export OMP_DISPLAY_ENV=verbose
# export OMP_DISPLAY_AFFINITY=true

if [[ $# == 0 ]]; then
  echo "Empty argument list, at least configuration file is required"
  usage
fi

# Clearing shared memory allocated by `PetscShmgetAllocateArray()`
ipcrm --all

$PETSC_DIR/lib/petsc/bin/petscfreesharedmemory

# This can be useful to track down `KSPSolve()` residues
# -predict_ksp_monitor_true_residual
# -correct_ksp_monitor_true_residual

# With `KSPSetReusePreconditioner()` the default "ilu" is usable!
# -predict_pc_type none
# -mpi_linear_solver_server
# -mpi_linear_solver_server_view

$MPI_DIR/bin/mpiexec                \
    -n $MPI_NUM_PROC                \
    ./build/xpic.out $@             \
