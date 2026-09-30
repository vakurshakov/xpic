---
name: building
description: Guidelines for building, running, testing, and formatting xpic.
---

# Building, running, and testing

Use this skill when working with the build system, CMake configuration, compilation issues, running the simulator, or the test suite.

## Environment

Environment variables (`MPI_DIR`, `JSON_DIR`, `PETSC_DIR`) are set in `header.sh` and sourced by all top-level scripts. Third-party dependencies (nlohmann::json, PETSc) are expected as sibling clones under `./external/` — see README.md for the exact clone/configure commands, including the two required PETSc arches: `linux-mpi-debug` and `linux-mpi-opt`.

## Top-level scripts

- `./build.sh` — configures and builds both `build/Debug` and `build/Release` with CMake (pass `Debug` or `Release` to build only one; parallel build with 4 jobs). Prefer `./build.sh Debug` for quick iteration and `./build.sh Release` (or no argument, which builds both) before running tests or benchmarking.
- `./run.sh <config.json> [petsc/mpi options]` — runs `./build/Release/xpic.out` via `mpiexec`, with OpenMP env vars set (`OMP_NUM_THREADS=4`, thread spread/affinity). Must be run from the repo root since output paths in configs are relative to it. Clears PETSc shared memory before each run.
- `./run-tests.sh [ctest args]` — rebuilds Release, then runs `ctest --test-dir build/Release`. Supports normal ctest filtering, e.g.:
  - `./run-tests.sh -R basic_ex1` — run a specific test by name.
  - `./run-tests.sh -L "impls ecsim"` — run tests by label (labels combine an area label like `impls`/`diagnostics`/`boris`/`crank_nicolson`/`drift_kinetic` with an implementation label like `basic`/`ecsim`/`ecsimcorr`/`eccapfim`, and MPI variants add the `mpi` label).
  - `./run-tests.sh --help` — shows ctest's own help.
- `./run-clang-format.sh` — formats all `.h`/`.cpp` files under `src/` and `tests/` in place with `clang-format` (config in `.clang-format`).

## Test layout

Tests are plain executables registered with `add_test` in each `tests/<area>/CMakeLists.txt`; most are added twice — once serial, once as an `mpiexec -np 2 ... -da_processors_x 2` MPI variant. When adding a new implementation or algorithm, add matching tests in the corresponding `tests/<area>/` directory and register them the same way, including both the serial and MPI variants.

## Build configuration

- CMake targets two configurations: `build/Debug` and `build/Release`. Debug is for development/debugging (fast iteration, symbols, no aggressive optimization); Release is what `run.sh` and `run-tests.sh` use.
- Required warning flags: `-Wall -Wextra -Wpedantic -Werror` — the build must be warning-clean; do not suppress warnings locally instead of fixing the underlying issue.
- C++ standard is C++20.
- PETSc is required and configured via `PETSC_DIR`/`PETSC_ARCH`; the two arches `linux-mpi-debug` and `linux-mpi-opt` correspond to the Debug/Release CMake configurations.
- MPI is required (code is built and tested with `mpiexec`, both `-np 1` and `-np 2`).

## Static analysis and formatting

- **clang-format**: run `./run-clang-format.sh` before committing; config lives in `.clang-format`. Header guards use `SRC_<PATH>_H` (not `#pragma once`).
- Keep CMake configuration in `CMakeLists.txt` files simple and readable — avoid complex generator expressions, deep custom macros, or conditional compilation for minor variations unless there's a strong reason.

## Dependency management

Keep external dependencies minimal. `xpic` depends on CMake, PETSc, MPI, and nlohmann::json. Do not add new external dependencies without strong justification — they add compilation complexity, version-compatibility issues, and maintenance burden.

## Configuration file format

`xpic` uses JSON for configuration, driven entirely by a config file passed as `./run.sh <config.json>` — there is no interactive mode. See `doc/src/commands.rst` for the JSON schema of commands/diagnostics, and `doc/src/structure.rst` for the broader config-driven architecture. Config-driven object construction matches a `name`/`command`/`diagnostic`/`Simulation` string field against a fixed set of literals in a builder or `build_*()` function; unknown values throw `std::runtime_error`.

## Troubleshooting builds

Common issues:
1. **CMake can't find PETSc**: verify `PETSC_DIR` and `PETSC_ARCH` are set correctly (sourced from `header.sh`).
2. **MPI compiler not found**: ensure `MPI_DIR` is set and the MPI compiler wrappers are on `PATH`.
3. **Linker errors**: verify `target_link_libraries` includes all needed libraries for the target.
4. **Stale build directory**: if CMake configuration changed significantly, remove `build/Debug` or `build/Release` and rerun `./build.sh`.
