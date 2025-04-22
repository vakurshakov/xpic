## Installation guide

#### Clone the repository

```sh
git clone https://github.com/vakurshakov/xpic.git
```

#### 1. Install [nlohmann::json](https://github.com/nlohmann/json)
```sh
git clone https://github.com/nlohmann/json.git ./external/json
```

#### 2. Install [PETSc](https://gitlab.com/petsc/petsc)
This is the short summary of PETSc installation process. To take an in-depth look on the configuring process, check the [documentation](https://petsc.org/release/install/install/) or run `./configure --help`. First, clone and update PETSc repository.

```sh
git clone -b release https://gitlab.com/petsc/petsc.git ./external/petsc
```

Change the directory to `./external/petsc/` and configure the library. The following code creates two configurations of the library with use of preinstalled MPI compilers and downloads just BLAS/LAPACK to the output directories `PETSC_ARCH`.

```sh
./configure PETSC_ARCH=linux-mpi-debug  \
  --with-fc=0                           \
  --with-mpi-dir=/opt/mpich/            \
  --with-threadsafety=1                 \
  --with-openmp=1                       \
  ---with-openmp-kernels=true           \
  --download-f2cblaslapack;             \
make PETSC_ARCH=linux-mpi-debug all;    \
make PETSC_ARCH=linux-mpi-debug check
```
```sh
./configure PETSC_ARCH=linux-mpi-opt             \
  --with-fc=0                                    \
  --with-mpi-dir=/opt/mpich/                     \
  --with-threadsafety=1                          \
  --with-openmp=1                                \
  ---with-openmp-kernels=true                    \
  --download-f2cblaslapack                       \
  --with-debugging=0                             \
  COPTFLAGS='-O3 -march=native -mtune=native'    \
  CXXOPTFLAGS='-O3 -march=native -mtune=native'; \
make PETSC_ARCH=linux-mpi-opt all;               \
make PETSC_ARCH=linux-mpi-opt check
```

If configure cannot automatically download the package, you can use a pre-downloaded one. Once the tarfile is downloaded, the path to this file can be specified to configure and it will proceed to install this package and then configure PETSc with this package. For example, one can download `f2cblaslapack` package locally and pass the configuration option `--download-f2cblaslapack=/path/to/local/f2cblaslapack-X.Y.Z.q.tar.gz`

#### 3. Compiling and running `xpic`

Now, the executable can be built successfully. To do so, run the following command from the home directory:
```sh
./build.sh
```

The binary will be created in the `./build` folder. Execution of the code should be performed from the home directory too:
```sh
./run.sh <config.json> [options]
```
