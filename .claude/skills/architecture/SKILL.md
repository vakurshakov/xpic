---
name: architecture
description: Understand xpic's execution flow, config-driven dispatch pattern, and core module responsibilities. Use before making cross-cutting changes or tracing how a simulation runs end to end.
---

# Architecture

Execution starts in `src/main.cpp`: PETSc is initialized, `Configuration::init(argv[1])` loads the JSON config into a global singleton (`CONFIG()`), `build_simulation()` picks a concrete simulation from the `"Simulation"` config field, then `initialize()` / `calculate()` / `finalize()` are called on it (Template Method pattern, base class `interfaces::Simulation` in `src/interfaces/simulation.h`).

## Config-driven dispatch pattern

This project makes heavy use of string-keyed factories driven by the JSON config: a JSON object has a `name`/`command`/`diagnostic`/`Simulation` field that is matched against a fixed set of string literals in a builder or `build_*()` function; unknown values throw `std::runtime_error`. When adding a new variant, find the existing `if (name == "X") ... else if (name == "Y")` chain for that concept and extend it, plus register the corresponding builder in the matching `builders/` subdirectory.

Errors thrown during `Builder::build()` are caught and re-augmented with the builder's `usage_message()` before rethrowing.

## `src/interfaces/`

Abstract base classes that every config-driven family implements against:

- `Simulation` (`simulation.h`/`.cpp`/`.tpp`) — the main-loop Template Method: `initialize()`, `calculate()`, `finalize()`.
- `Particles` (`particles.h`/`.cpp`) — per-species particle container/behavior contract implemented by each `src/impls/<scheme>/`.
- `Command` (`command.h`) — `execute(timestep)`, implemented by everything in `src/commands/`.
- `Diagnostic` (`diagnostic.h`/`.cpp`) — `diagnose(timestep)`, implemented by everything in `src/diagnostics/`.
- `Builder` (`builder.h`/`.cpp`) — factory/builder base with JSON parsing helpers for vectors, geometries, etc., shared by `commands/builders/` and `diagnostics/builders/`.
- `Point` (`point.h`/`.cpp`) and `SortParameters` (`sort_parameters.h`/`.cpp`) — shared value types (grid point, particle-sort configuration) used across schemes.

## `src/impls/<scheme>/`

Concrete `Simulation` + `Particles` implementations, one pair per numerical scheme, selected by the `"Simulation"` config key. Each directory should contain both `particles.{h,cpp}` and `simulation.{h,cpp}`.

- `basic/` — explicit scheme: Boris push, Esirkepov current deposition, FDTD field update.
- `ecsim/` — energy-conserving semi-implicit scheme.
- `ecsimcorr/` — energy- *and* charge-conserving scheme; extends `ecsim/`.
- `eccapfim/` — fully implicit scheme conserving both energy and charge.

Schemes differ mainly in how the particle push and field solve are coupled (explicit vs. semi-implicit vs. fully implicit); the shared numerical primitives they call into live in `src/algorithms/`, not duplicated per scheme.

## `src/algorithms/`

Numerical building blocks shared across `src/impls/` schemes, with no config-dispatch layer of their own — they are called directly by the scheme that needs them:

- Particle pushers: `boris_push`, `crank_nicolson_push`, `drift_kinetic_push` / `drift_kinetic_implicit`, `ricketson_push`.
- Charge/current deposition: `simple_decomposition`, `esirkepov_decomposition`, `implicit_esirkepov`.
- Field interpolation: `simple_interpolation`.

## `src/commands/` + `src/commands/builders/`

Actions that modify the simulation outside the main physics step: `InjectParticles`, `SetParticles`, `RemoveParticles`, `SetMagneticField`, `FieldsDamping`. Config `"Presets"` run once at init; `"StepPresets"` run once per timestep. Each command has a matching builder in `builders/` (`command_builder.h`/`.cpp` holds the dispatch chain). See `doc/src/commands.rst` for the full JSON schema of each command, including shared `"coordinate"`/`"momentum"`/`"geometry"` sub-config formats (`PreciseCoordinate`/`CoordinateInBox`/`CoordinateInCylinder`, `PreciseMomentum`/`MaxwellianMomentum`, `BoxGeometry`/`CylinderGeometry`).

## `src/diagnostics/` + `src/diagnostics/builders/`

Output/logging of simulation data, run on the config's `diagnose_period`, writing to `OutputDirectory`:

- `FieldView` / `FieldViewZAvg` — raw and Z-averaged field snapshots.
- `DistributionMoment` / `VelocityDistribution` — particle-distribution moments and velocity-space histograms.
- `Energy`, `ChargeConservation`, `MomentumConservation` — conservation checks, most relevant to `ecsim`/`ecsimcorr`/`eccapfim`.
- `LogView` — periodic scalar log output.
- `MatDump` — raw PETSc matrix dump, mainly for debugging implicit solves.
- `TableDiagnostic` — base helper for tabular (CSV-like) diagnostics; not config-dispatched directly.
- `SimulationBackup` — checkpoint/restart state.

Not every diagnostic has a builder (e.g. `charge_conservation`, `energy`, `momentum_conservation`, `mat_dump`, `table_diagnostic` are built inline or subclassed rather than through `diagnostics/builders/`); only the ones with matching files in `builders/` are exposed as their own config-dispatch entries — check `diagnostic_builder.cpp` for the authoritative list before assuming a diagnostic is config-selectable.

## `src/utils/`

Cross-cutting infrastructure with no config-dispatch layer:

- `Configuration` (`configuration.h`/`.cpp`) — JSON config loading, accessed globally via `CONFIG()`.
- `World` (`world.h`/`.cpp`) — grid/world geometry (domain size, spacing, PETSc DMDA setup).
- `Geometries` (`geometries.h`/`.cpp`), `Shape` (`shape.h`/`.cpp`) — geometric primitives used by commands/particle loading.
- `ParticlesLoad` (`particles_load.h`/`.cpp`) — initial particle loading/seeding.
- `Vector3`/`Vector4`/`vector_utils.h`, `Operators` (`operators.h`/`.cpp`) — math/vector helpers.
- `TableFunction` (`table_function.h`/`.cpp`) — tabulated function evaluation (e.g. for arbitrary profiles).
- `RandomGenerator` (`random_generator.h`), `ZigguratGaussian` (`ziggurat_gaussian.h`/`.cpp`) — RNG, including Maxwellian sampling.
- `SyncClock` (`sync_clock.h`/`.cpp`) — timing/profiling instrumentation.
- `utils.h`/`.cpp` — the `LOG(...)`, `LOG_IMPL(...)`, `LOG_FLUSH()` logging macros (format-string style, similar to `std::format`) and other small helpers.

## `doc/`

Sphinx docs (Russian). `doc/src/structure.rst` covers the architecture in more depth; `doc/src/commands.rst` documents command JSON schemas; `doc/src/installation.rst` covers build/install.

## `tools/`

Standalone Python scripts, not part of the C++ build: `basic_run.py` / `basic_ffmpeg.py` for driving batches of simulations, `tools/plot/` and `tools/lib/` for plotting/post-processing diagnostic output, `tools/collect/` for gathering results across runs.

---

Always verify path existence before suggesting changes (do not rely on stale memory).
