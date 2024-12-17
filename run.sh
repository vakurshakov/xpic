#!/bin/bash

usage() { echo "Usage: $0 <config.json>" 1>&2; exit 1; }

export MPI_NUM_PROC=1

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREAD=1

# export OMP_DISPLAY_ENV=verbose
# export OMP_DISPLAY_AFFINITY=true  # to measure thread migration

if [[ $# == 0 ]]; then
  echo "Empty argument list, at least configuration file is required"
  usage
fi

mpiexec -np $MPI_NUM_PROC ./build/xpic.out $1
