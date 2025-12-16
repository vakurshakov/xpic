#!/bin/bash

source ./header.sh

build_type=Release

# source ./build.sh $build_type

# see https://github.com/pmodels/mpich/issues/4616
# see https://github.com/pmodels/mpich/blob/main/doc/wiki/how_to/Using_the_Hydra_Process_Manager.md#process-core-binding
export HYDRA_BINDING=none
export HYDRA_MAPPING=none

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=spread # see https://www.openmp.org/spec-html/5.0/openmpse52.html
export OMP_PLACES=threads   # see https://www.openmp.org/spec-html/5.0/openmpse53.html

# Can help to inspect thread migration
export OMP_DISPLAY_ENV=false       # [true | false | verbose], see https://www.openmp.org/spec-html/5.0/openmpse60.html
export OMP_DISPLAY_AFFINITY=false  # [true | false],           see https://www.openmp.org/spec-html/5.0/openmpse61.html

# Clearing shared memory allocated by `PetscShmgetAllocateArray()`
ipcrm --all
$PETSC_DIR/lib/petsc/bin/petscfreesharedmemory

MPI_NUM_PROC=1

# Running the main executable, namely `xpic.out` 
$MPI_DIR/bin/mpiexec -n $MPI_NUM_PROC ./build/$build_type/xpic.out $@ \
    # -da_processors_x 2 \
    # -da_processors_y 2 \
    # -da_processors_z 2 \