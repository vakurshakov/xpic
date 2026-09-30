---
name: directory-structure
description: Understand xpic code organization and locate components when exploring the repository or deciding where new code belongs.
---

# Instructions

* Use when exploring the repository, trying to find a component, or deciding where new code belongs.
* Operate relative to the repository root (`git rev-parse --show-toplevel`).
* Always verify path existence before suggesting changes (do not rely on stale memory).

# `xpic` file structure

`xpic` is a C++ particle-in-cell (PIC) plasma physics simulation code built on PETSc (MPI + OpenMP parallelism) with nlohmann::json for configuration.

## Top-level structure

```
src/
├── interfaces/     - Abstract base classes and interfaces
├── impls/          - Concrete simulation + particles implementations, one pair per numerical scheme
├── commands/       - Config-driven actions that modify the simulation outside the main physics step
├── algorithms/     - Numerical building blocks shared across implementations (pushers, current decomposition)
├── diagnostics/    - Output/logging of simulation data
└── utils/          - Config loading, world/grid geometry, math/vector helpers, particle loading, RNG

tests/             - Test executables, one CMakeLists.txt per area, mirroring src/ areas

tools/             - Standalone Python scripts (not part of the C++ build): batch runs, plotting/post-processing

doc/               - Sphinx docs (Russian); structure.rst and commands.rst are the key references

CMakeLists.txt, build.sh, header.sh, run.sh, run-tests.sh, run-clang-format.sh, .clang-format - Project config/scripts
```

For execution flow, the config-driven dispatch pattern, per-directory responsibilities, and the module dependency hierarchy, see `.claude/skills/architecture/SKILL.md`.

## Search hierarchy when looking for code

1. **User-specified path**: if the user named a directory or file, start there.
2. **Interfaces** (`src/interfaces/`): if defining a new component type (e.g. a new pusher interface), start here.
3. **Impls** (`src/impls/<scheme>/`): if implementing or modifying a specific numerical scheme.
4. **Algorithms** (`src/algorithms/`): if adding/modifying a pusher or current-decomposition method shared across schemes.
5. **Commands** (`src/commands/`): if adding a config-driven action that runs as a `Preset`/`StepPreset`.
6. **Diagnostics** (`src/diagnostics/`): if adding new simulation output/logging.
7. **Utils** (`src/utils/`): if adding config, geometry, math, or RNG utilities.
8. **Tests** (`tests/<area>/`): tests live per area, mirroring the `src/` layout; see the `building` skill for how tests are registered and run.

## Example: adding a new particle pusher

1. Add the pusher implementation in `src/algorithms/` (pushers are shared building blocks, not per-scheme).
2. Wire it into the relevant `src/impls/<scheme>/` implementation(s) that should use it.
3. Add tests in the matching `tests/<area>/` directory (serial + MPI variant, see the `building` skill).
4. Expose it via config if it should be selectable, following the existing string-keyed dispatch pattern.
5. Add diagnostics in `src/diagnostics/` only if the new pusher needs special output.

## External dependencies

`xpic` depends on CMake, PETSc, MPI, and nlohmann::json. Do not add new external dependencies without strong justification — they add complexity and maintenance burden.
