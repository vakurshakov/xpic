# `xpic`

Particle-in-cell plasma simulation code built on PETSc (MPI + OpenMP) and nlohmann::json. A run is driven entirely by a JSON config file. All commands below are run from the repository root.

```sh
git clone https://github.com/vakurshakov/xpic.git
cd xpic
```

## Dependencies

| Dependency | Minimum | Tested | Installed by `install-deps.sh` |
| --- | --- | --- | --- |
| git | — | — | system package manager |
| C++20 compiler (`gcc`, `g++`) | unverified | GCC 15.1, GCC 16.2, AOCC 5.2.0 (with `GCC_DIR`) | system package manager |
| `gfortran` (local only, OpenBLAS build needs it) | — | GCC 15.1 | system package manager |
| MPI (`mpicxx`) | unverified | Open MPI 5.0.11 (local), MPICH 5.0.2 (stark) | system package manager, never built from source |
| make | — | — | system package manager |
| CMake | 3.10 | 4.4 | system package manager |
| nlohmann::json | 3.11 | v3.11.2 | cloned into `./external/json` |
| BLAS/LAPACK | — | OpenBLAS 0.3.34 (local), f2cblaslapack and AOCL 5.3.0 (stark) | through PETSc (`--download-openblas`) |
| PETSc | 3.26 | v3.26.1 | cloned and built into `./external/petsc` |
| Python 3 | 3.6 | 3.14 | system package manager |
| numpy, matplotlib, scipy | — | — | optional, only for post-processing in `tools/` |

The source of truth is [scripts/versions.sh](scripts/versions.sh).

## 1. Install dependencies

```sh
./scripts/install-deps.sh
```

The script checks `git`, `gcc`, `g++`, `gfortran`, `mpicxx`, `make`, `cmake` and `python3` one by one and offers the package-manager command (`pacman`, `apt-get`, `dnf`, `brew`) for each missing one. Then it clones nlohmann::json and PETSc at the pinned tags into `./external` and builds both PETSc arches (`linux-mpi-debug`, `linux-mpi-opt`), running `make check` after each.

Options:

- `--install` accepts all package-manager prompts (unattended setups, containers);
- `debug` or `opt` builds a single PETSc arch;
- `--force` re-fetches and rebuilds even if the pinned version is present.

Re-running the script is a no-op for dependencies that are already built at the pinned version.

## 2. Build

```sh
./scripts/build.sh            # Debug and Release
./scripts/build.sh Release    # only one of them
```

The executables are `build/Debug/xpic.out` and `build/Release/xpic.out`. The build fails early if nlohmann::json or PETSc is missing or older than the minimum.

## 3. Test

```sh
./scripts/run-tests.sh -L smoke
```

Arguments are passed to `ctest`; levels are `smoke` (fast, one mechanism), `sanity` (short run, one invariant) and `regression` (long runs). Add `-LE mpi` to skip the MPI variants.

## 4. Run

```sh
./scripts/run.sh examples/basic.json
```

This runs `build/Release/xpic.out` through `mpiexec` and writes the output to `results/basic`. Everything after the config path is forwarded to `xpic.out` (PETSc options included). Tune the launch with `MPI_NUM_PROC` (default 1), `OMP_NUM_THREADS` (default 4), `OMP_PROC_BIND`, `OMP_PLACES` and `RUN_PREFIX` (a command inserted before the binary, e.g. `numactl --localalloc`):

```sh
MPI_NUM_PROC=2 OMP_NUM_THREADS=2 ./scripts/run.sh examples/basic.json -da_processors_x 2
```

## Environment overrides and `STACK`

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
