# `xpic`

Particle-in-cell plasma simulation code built on PETSc (MPI + OpenMP) and nlohmann::json. A run is driven entirely by a JSON config file. All commands below are run from the repository root.

```sh
git clone https://github.com/vakurshakov/xpic.git
cd xpic
```

## Dependencies

Two ways to get them: **local** (a workstation, [Local setup](#local-setup)) and **stark** (the offline SGE cluster, [Cluster setup](#cluster-setup-stark)).

| Dependency | Minimum | Tested | Local | stark |
| --- | --- | --- | --- | --- |
| git | 1.8.5 | 2.56.0 | system package manager | not needed (the repository is copied with `rsync`) |
| C++20 compiler (`gcc`, `g++`) | unverified | GCC 15.1, GCC 16.2, AOCC 5.2.0 (with `GCC_DIR`) | system package manager | GCC built by `install-gcc.sh`; AOCC is a manual vendor install, and still needs that GCC (`GCC_DIR`) |
| `gfortran` | — | GCC 15.1 | system package manager (OpenBLAS build needs it) | not needed (no Fortran) |
| MPI (`mpicxx`) | unverified | Open MPI 5.0.11 (local), MPICH 5.0.2 (stark) | system package manager, never built from source |`install-mpich.sh` |
| make | — | — | system package manager | system |
| CMake | 3.10 | 4.4 | system package manager | system |
| nlohmann::json | 3.11 | v3.12.0 | cloned into `./external/json` |`install-json.sh` |
| BLAS/LAPACK | — | OpenBLAS 0.3.34 (local), f2cblaslapack and AOCL 5.3.0 (stark) | through PETSc (`--download-openblas`) | OpenBLAS (`install-openblas.sh`), f2cblaslapack (through PETSc, from a tarball), or AOCL (manual vendor install), by stack |
| PETSc | 3.26 | v3.26.1 | cloned and built into `./external/petsc` | `install-petsc.sh` |
| Python 3 | 3.6 | 3.14 | system package manager | `anaconda3`, installation is not described |
| numpy, matplotlib, scipy | — | — | install yourself: needed by the Python post-processing in `tools/` and by the `*_plot` tests (they validate the generated data) | same |

The source of truth is [scripts/versions.sh](scripts/versions.sh).

## Local setup

### 1. Install dependencies

```sh
./scripts/install-deps.sh
```

The script checks `git`, `gcc`, `g++`, `gfortran`, `mpicxx`, `make`, `cmake`, `python3` and `diff` one by one and offers the package-manager command (`pacman`, `apt-get`, `dnf`, `brew`) for each missing one. Then it clones nlohmann::json and PETSc at the pinned tags into `./external` and builds both PETSc arches (`linux-mpi-debug`, `linux-mpi-opt`), running `make check` after each.

Options:

- `--install` accepts all package-manager prompts (unattended setups, containers);
- `debug` or `opt` builds a single PETSc arch;
- `--force` re-fetches and rebuilds even if the pinned version is present.

Every script prints its usage, options and the environment variables it reads with `--help` (e.g. `./scripts/install-deps.sh --help`).

Re-running the script is a no-op for dependencies that are already built at the pinned version.

### 2. Build

```sh
./scripts/build.sh            # Debug and Release
./scripts/build.sh Release    # only one of them
```

The executables are `build/Debug/xpic.out` and `build/Release/xpic.out`. The build fails early if nlohmann::json or PETSc is missing or older than the minimum.

### 3. Test

```sh
./scripts/run-tests.sh -L smoke
```

Arguments are passed to `ctest`; levels are `smoke` (fast, one mechanism), `sanity` (short run, one invariant) and `regression` (long runs). Add `-LE mpi` to skip the MPI variants.

### 4. Run

```sh
./scripts/run.sh examples/basic.json
```

This runs `build/Release/xpic.out` through `mpiexec` and writes the output to `results/basic`. Everything after the config path is forwarded to `xpic.out` (PETSc options included). Tune the launch with `MPI_NUM_PROC` (default 1), `OMP_NUM_THREADS` (default 4), `OMP_PROC_BIND`, `OMP_PLACES` and `RUN_PREFIX` (a command inserted before the binary, e.g. `numactl --localalloc`):

```sh
MPI_NUM_PROC=2 OMP_NUM_THREADS=2 ./scripts/run.sh examples/basic.json -da_processors_x 2
```

### Environment overrides and `STACK`

All scripts source [scripts/setenv.sh](scripts/setenv.sh), the only place where dependency locations are decided. Every variable can be overridden from the environment, inline on the command:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MPI_DIR` | parent of the `mpicxx` found on `PATH` | MPI installation; its wrapper defines the compiler |
| `JSON_DIR` | `./external/json` | nlohmann::json checkout |
| `PETSC_DIR` | `./external/petsc` | PETSc installation |
| `PETSC_ARCH` | `linux-mpi-debug` / `linux-mpi-opt` for Debug / Release | PETSc arch |
| `GCC_DIR` | unset | GCC toolchain for Clang-based stacks (libstdc++ and rpath) |

A `PETSC_DIR` or `JSON_DIR` outside `./external` is never cloned or built, only verified: `install-deps.sh` checks the version header against the minimum and, for PETSc, that the library is built; `build.sh` re-checks the versions on every run and fails if they are too old.

```sh
PETSC_DIR=/opt/petsc PETSC_ARCH=arch-opt ./scripts/build.sh Release
```

`STACK=<name>` sources `scripts/stacks/<name>.sh` before the defaults are applied, which lets a machine set these variables once. The default is `local`; an unknown name fails.

```sh
STACK=local ./scripts/build.sh Release
```

## Cluster setup (stark)

stark is an offline SGE cluster: the scripts in [scripts/cluster/](scripts/cluster) build the whole stack from source tarballs, never download anything, and write into one directory tree, `STACK_ROOT`. Everything below is a template: replace `/path/to/...`, `<queue>` and the stack name. All commands run from the repository root; the install scripts source `./scripts/...` and `qsub` uses `-cwd`.

### 1. Prepare

Fetch the tarballs on a machine with internet and copy them into one directory, `TARBALL_DIR`:

| Tarball | Source | Needed by |
| --- | --- | --- |
| `gcc-15.1.0.tar.gz` | [GCC releases](https://gcc.gnu.org/releases.html); GMP, MPFR and MPC must be on the host or unpacked into the sources beforehand (`contrib/download_prerequisites`) | `install-gcc.sh` |
| `mpich-5.0.2.tar.gz` | [MPICH downloads](https://www.mpich.org/downloads/) | `install-mpich.sh` |
| `OpenBLAS-0.3.34.tar.gz` | [OpenBLAS releases](https://github.com/OpenMathLib/OpenBLAS/releases) | `install-openblas.sh` (`stark-gcc-openblas`) |
| `f2cblaslapack-3.8.0.q2.tar.gz` | [PETSc mirror](https://web.cels.anl.gov/projects/petsc/download/externalpackages/) | PETSc, `stark-gcc-f2c` |
| `petsc-3.26.1.tar.gz` | [PETSc downloads](https://petsc.org/release/download/) | `install-petsc.sh` |
| `json-3.12.0.tar.gz` | [nlohmann/json releases](https://github.com/nlohmann/json/releases) (source of tag `v3.12.0`) | `install-json.sh` |
| `aocc-compiler-5.2.0.tar`, `aocl-linux-aocc-5.3.0.tar.gz` | [AMD AOCC](https://www.amd.com/en/developer/aocc.html), [AMD AOCL](https://www.amd.com/en/developer/aocl.html) | `stark-aocc-aocl`, manual install (step 4) |

Copy the repository to the cluster (no git there) and set the variables once per shell:

```sh
git ls-files --exclude-standard | rsync -a --files-from=- ./ stark:/path/to/xpic/
ssh stark
cd /path/to/xpic
export STACK_ROOT=/path/to/stack/root   # root for all of the components
export TARBALL_DIR=/path/to/tarballs    # common place for component tarballs
export STACK=stark-gcc-openblas         # or stark-gcc-f2c, stark-aocc-aocl
```

Tarball names are declared in the install scripts (the f2cblaslapack one in `scripts/stacks/stark-gcc-f2c.sh`), versions in [scripts/versions.sh](scripts/versions.sh).

### 2. Choose a stack

| `STACK` | Compiler | BLAS/LAPACK | `PETSC_DIR` under `STACK_ROOT` |
| --- | --- | --- | --- |
| `stark-gcc-openblas` | GCC | OpenBLAS | `petsc-3.26.1/GCC-15.1-OpenBLAS` |
| `stark-gcc-f2c` | GCC | f2cblaslapack only (BLAS/LAPACK translated to C; no OpenBLAS, no Fortran) | `petsc-3.26.1/GCC-15.1-F2C` |
| `stark-aocc-aocl` | AOCC (Clang) | AOCL (BLIS + libFLAME) | `petsc-3.26.1/AOCC-5.2.0-AOCL` |

Every variable of a stack can still be overridden from the environment, e.g. `GCC_DIR=/path/to/GCC` when the GCC is not under `STACK_ROOT`. A stack fails early without `STACK_ROOT`.

### 3. Install, in this order

Build on a node of the target architecture: the stacks use `-march=native`, a library built on a login node may crash on a compute node of another CPU. Each script prints its usage with `--help`; run it directly or submit it (queue, slots and runtime go on the command line):

```sh
# GCC first: it is the compiler of the gcc stacks and the libstdc++ of AOCC
qsub -q <queue> -pe smp <n> -v STACK,STACK_ROOT,TARBALL_DIR ./scripts/cluster/install-gcc.sh

# wait for the job to finish (qstat), then
qsub -q <queue> -pe smp <n> -v STACK,STACK_ROOT,TARBALL_DIR ./scripts/cluster/install-mpich.sh

# stark-gcc-openblas only: OpenBLAS (stark-gcc-f2c and stark-aocc-aocl skip it)
qsub -q <queue> -pe smp <n> -v STACK,STACK_ROOT,TARBALL_DIR ./scripts/cluster/install-openblas.sh

# wait for the job to finish (qstat), then
qsub -q <queue> -pe smp <n> -v STACK,STACK_ROOT,TARBALL_DIR ./scripts/cluster/install-petsc.sh

# header-only, nothing to compile: run it directly
./scripts/cluster/install-json.sh
```

For `stark-aocc-aocl` do step 4 before `install-petsc.sh` (and before MPICH: it is built with AOCC). Each step needs the previous ones installed. Job logs are `install-<component>.o<jobid>` in the current directory. If a job is queued on a busy node, `qalter <jobid> -q <queue>@<node>` moves it.

Options and rules, the same for all five scripts:

- `--force` rebuilds an existing prefix from a clean source tree; without it an existing prefix is refused.
- `JOBS` sets the `make` parallelism (default: scheduler slots, then `nproc`).
- Out-of-tree build directories (`build-<stack>`) are removed at the start of every run.

### 4. AOCC and AOCL (`stark-aocc-aocl` only)

These are vendor installs with an EULA; the scripts only expect them in place. Unpack each archive so that its `install.sh` ends up directly in `aocc-compiler-5.2.0/` and `AOCL-5.3.0/` under `STACK_ROOT` (read the vendor `README`; `tar --strip-components=1` removes a redundant top-level directory), then run the installers:

```sh
(cd "$STACK_ROOT/aocc-compiler-5.2.0" && bash install.sh)
(cd "$STACK_ROOT/AOCL-5.3.0" && bash install.sh -t .)
```

The AOCC installer only creates links and profiles. The AOCL installer is interactive:

- the EULA: page to the end with `Space`, type `yes`;
- "LP64 or ILP64 libraries as default?": **LP64**. ILP64 uses 64-bit integers and does not work with PETSc;
- afterwards `AOCL-5.3.0/MT/lib` and `ST/lib` must exist (the stack links the multithreaded `MT`).

### 5. Build and run `xpic`

```sh
./scripts/build.sh Release   # same STACK, STACK_ROOT as above
./scripts/run.sh examples/basic.json
```

For a batch run, wrap `./scripts/run.sh` in a job script and pass `STACK` and `STACK_ROOT` as above. The stack sets `LD_LIBRARY_PATH` to its own libraries only, so an inherited Anaconda (`libstdc++`, `libmpifort`) cannot shadow them.

### Layout on disk

```text
$STACK_ROOT/
  GCC-15.1/              GCC prefix; used by AOCC compiler
    gcc-15.1.0/          source, shared by the stacks
    build-<stack>/       out-of-tree build
  aocc-compiler-5.2.0/   AOCC (manual)
  AOCL-5.3.0/            AOCL (manual); MT/ is linked
    ST/                  single-threaded libraries
    MT/                  multi-threaded libraries
  MPICH-5.0.2/
    mpich-5.0.2/         source, shared by the stacks
    build-<stack>/       out-of-tree build
    <compiler>/          MPI_DIR
  OpenBLAS-0.3.34/
    OpenBLAS-0.3.34/     source, built in place
    <compiler>/          OPENBLAS_DIR
  petsc-3.26.1/
    petsc-3.26.1/        source, the PETSC_ARCH build directory is inside
    <compiler>-<BLAS>/   PETSC_DIR
  json-3.12.0/           source == JSON_DIR (header only library)
```

### Known sharp edges

- The directory a script uses (`NAME`) repeats the directory part of the stack's `*_DIR`; if a stack overrides a `*_DIR`, the source still goes to `$STACK_ROOT/<NAME>`.
- If `GCC-15.1` in `STACK_ROOT` is a symlink to another tree, a GCC build writes into that tree.
- OpenBLAS and PETSc share the source tree between stacks: `--force` on one wipes the source for all of them (installed prefixes are not affected).
- Clang (AOCC) takes the legacy system GCC by default, which cannot build C++20; that is why the AOCC stack passes `--gcc-toolchain=$GCC_DIR` to both compile and link.
