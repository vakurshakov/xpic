#!/bin/bash

source ./header.sh

usage() { echo "Usage: $0 <config.json>" 1>&2; exit 1; }

MPI_NUM_PROC=1
HYDRA_BINDING=none
HYDRA_MAPPING=none

OMP_NUM_THREADS=1
OMP_PLACES=cores
OMP_PROC_BIND=spread

while read -r line; do
  case $line in
    *num_threads*)
      line=${line#*: }
      line=${line%,}
      OMP_NUM_THREADS=$line
      ;;
    *da_processors*)
      line=${line#*: }
      line=${line%,}
      MPI_NUM_PROC=$(( $MPI_NUM_PROC * $line ))
      ;;
  esac
done < config.json

export MPI_NUM_PROC
export HYDRA_BINDING
export HYDRA_MAPPING

export OMP_NUM_THREADS
export OMP_PLACES
export OMP_PROC_BIND

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
