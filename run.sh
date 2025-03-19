#!/bin/bash

source ./header.sh

if [[ $# == 0 ]]; then
  echo "Empty argument list, at least configuration file is required"
  echo "Usage: $0 <config.json>" 1>&2;
  exit 1;
fi

MPI_NUM_PROC=1

# see https://github.com/pmodels/mpich/issues/4616
# see https://github.com/pmodels/mpich/blob/main/doc/wiki/how_to/Using_the_Hydra_Process_Manager.md#process-core-binding
HYDRA_BINDING=none
HYDRA_MAPPING=none

OMP_NUM_THREADS=1
OMP_PROC_BIND=true # see https://www.openmp.org/spec-html/5.0/openmpse52.html
OMP_PLACES=cores   # see https://www.openmp.org/spec-html/5.0/openmpse53.html

#                           # Can help to inspect thread migration
OMP_DISPLAY_ENV=false       # [true | false | verbose], see https://www.openmp.org/spec-html/5.0/openmpse60.html
OMP_DISPLAY_AFFINITY=false  # [true | false],           see https://www.openmp.org/spec-html/5.0/openmpse61.html

parse() {
  line=$(echo $1 | sed s/\"//g)
  line=${line#*: }
  line=${line%,}
  echo $line
}

# Reading mpi and omp configuration from config.json
while read line; do
  case $line in
    *da_processors*)
      MPI_NUM_PROC=$(( $MPI_NUM_PROC * $(parse "$line") )) ;;
    *num_proc*)
      MPI_NUM_PROC=$(parse "$line") ;;
    *binding*)
      HYDRA_BINDING=$(parse "$line") ;;
    *mapping*)
      HYDRA_MAPPING=$(parse "$line") ;;
    *num_threads*)
      OMP_NUM_THREADS=$(parse "$line") ;;
    *places*)
      OMP_PLACES=$(parse "$line") ;;
    *proc_bind*)
      OMP_PROC_BIND=$(parse "$line") ;;
    *display_env*)
      OMP_DISPLAY_ENV=$(parse "$line") ;;
    *display_affinity*)
      OMP_DISPLAY_AFFINITY=$(parse "$line") ;;
  esac
done < $1  # config.json

if (( MPI_NUM_PROC < 0 ))
then
  echo "Please, avoid the explicit use of negative 'da_processors*'."
  echo "If you want to use '-mpi_linear_solver_server', pass a 'num_proc' in your configuration file."
  exit 1
fi

export HYDRA_BINDING=$HYDRA_BINDING
export HYDRA_MAPPING=$HYDRA_MAPPING

export OMP_NUM_THREADS=$OMP_NUM_THREADS
export OMP_PLACES=$OMP_PLACES
export OMP_PROC_BIND=$OMP_PROC_BIND
export OMP_DISPLAY_ENV=$OMP_DISPLAY_ENV
export OMP_DISPLAY_AFFINITY=$OMP_DISPLAY_AFFINITY

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
