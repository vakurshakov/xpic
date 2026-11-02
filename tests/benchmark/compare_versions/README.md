# Comparing two xpic versions

Runs the same JSON config through two `xpic.out` builds and prints before/after tables (PETSc `-log_view`, `perf stat`, `perf record`). Used to close the "Still open" item of `.claude/plans/particle-storage-deferred.md`: the real pushers before and after the storage change (results below).

## Usage

```
python3 -I tests/benchmark/compare_versions/run.py <before> [<after>] [--repeats 3] [--sims basic,ecsim,ecsimcorr,eccapfim] [--keep] [--workdir DIR]
```

- `<before>`, `<after>`: an existing checkout directory (used as is, never removed) or a git revision (detached `git worktree` in the workdir, `external` symlinked, removed at the end unless `--keep`). `<after>` defaults to the current tree.
- Every checkout is built in `build/Release` (`header.sh`, `cmake`, target `xpic.out`). A failed build aborts and prints the tail of the log; old commits are not patched.
- Scratch: `<tempdir>/xpic_compare/` (`TMPDIR` is honoured), or `--workdir`.
- Output: `output/results.md` (header with both revisions, date, command) and `output/results.json`; `output/` is gitignored.

```
python3 -I tests/benchmark/compare_versions/run.py 9dee688a            # revision vs current tree
python3 -I tests/benchmark/compare_versions/run.py /path/to/checkout   # directory vs current tree
python3 -I tests/benchmark/compare_versions/run.py <rev_a> <rev_b>
```

Limits: the one config ([config.template.json](config.template.json)) must parse on both versions; `perf` needs `perf_event_paranoid <= 2`. The benchmark is wall-clock sensitive, do not build or run other load during it. A full run (all simulations, 3 repeats) takes about an hour.

## Setup

Periodic cube, warm Maxwellian electrons (`T=0.1`, `MaxwellianMomentum`, `tov`), background ions, as in `basic_ex4` / `ecsim_ex3` / `eccapfim_ex4`. Release, 4 threads in total (1 rank x 4, 2 ranks x 2, `OMP_PROC_BIND=spread`, `OMP_PLACES=threads`, Open MPI `--bind-to none --oversubscribe`), 3 repeats, before/after interleaved in each repeat, `mean ± sd`.

| simulation | `-n` | `-ppc` | particles | steps | dx | dt |
| ---------- | ---- | ------ | --------- | ----- | -- | -- |
| basic | 16 | 64 | 262144 | 20 | 0.05 | 0.025 |
| ecsim | 16 | 64 | 262144 | 20 | 0.5 | 1.5 |
| ecsimcorr | 16 | 64 | 262144 | 20 | 0.5 | 1.5 |
| eccapfim | 12 | 32 | 55296 | 3 | 0.5 | 1.5 |

eccapfim uses Anderson SNES (`atol = rtol = 1e-7`).

Measured: PETSc `-log_view` stages and events (summed over stages), `perf stat` (`cycles`, `instructions`, `L1-dcache-load-misses`, user space only; LLC events are unsupported with `perf_event_paranoid=2`), `perf record` (1 rank, self time). Values are ms/step or G(events)/step. If the final `wK` is identical in both versions (same RNG seed), the runs are equivalent.

Not a pure pusher benchmark: the per-step diagnostics (conservation checks, moments) run in every step and iterate the storage as well, they are the `Diagnostics run` stage. `wall` includes initialization.

# Result: list storage vs `ParticleStorage`

| | commit | storage |
| - | ------ | ------- |
| before | `9dee688a` (2026-10-28, "point.h was incorrectly removed from the particle_trace") | `std::vector<std::list<Point>>` |
| after | `cd7c40d8` ("changing mpiexec to use -np within codebase"), tip of `particle-storage` | `ParticleStorage` (variant `e`) |

`56da9272` (the spike commit, last one with the old storage) does not compile: `particle_trace.cpp` misses the `point.h` include. `9dee688a` adds only that include and the glossary, `src/` is otherwise identical.

## Summary

The new storage does not lose on any pusher, so there is no reason to fall back to `v`. The always-on diagnostics are 1.5-1.8x slower and eat part of the gain in the total time.

after/before, 1 rank x 4 threads:

| simulation | pusher stages | update_cells | diagnostics | wall | IPC |
| ---------- | ------------- | ------------ | ----------- | ---- | --- |
| basic | Main Stage 0.71 | 0.45 | **1.79** | 0.89 | 1.58 -> 2.08 |
| ecsim | First push 0.82, Second push 0.56 | 0.51 | **1.81** | **1.04** | 1.54 -> 1.75 |
| ecsimcorr | First push 0.64, Second push 0.60 | 0.48 | **1.83** | 0.83 | 1.35 -> 1.91 |
| eccapfim | Calc iteration 1.01 (noise) | 0.95 | 1.98 | 1.02 | 2.09 -> 2.12 |

- basic has no `first_push` / `second_push` events, the Main Stage stands in.
- ecsim, 2 ranks x 2 threads: wall 1.07 (diagnostics 1.73, First push 0.78, Second push 0.59). ecsimcorr, 2 ranks: wall 0.96 (First push 0.76, Second push 0.78).
- The biggest single event: `final_update` of ecsimcorr is 25x faster (8.4 -> 0.33 ms/step, 1 rank).
- Instructions are 15-30% higher after, cycles are lower or equal: the gain is IPC (L1 misses of basic and ecsimcorr drop by 25-55%).
- eccapfim: `form_iteration` / `prepare_storage` events do not exist in the old code (shown as `-`). The +1.5% is the timer overhead of the new instrumentation (`__vdso_clock_gettime` + `omp_get_wtime` are 5.7% of samples after), which matches the phase-timing driver that was used before (within 0-2.5%, see `.claude/plans/particle-storage-deferred.md`). The `perf record` profile shapes are the same: interpolation and shape functions dominate (> 65%).

**Regression to look at:** `Diagnostics run` is 1.7-1.8x slower after (basic 67 -> 119 ms/step, about +50 ms/step), 1.5-2.0x for eccapfim. In the profiles `MomentumConservation::calculate` goes from 3.7-9.6% to 8.7-13.4% of the samples. It makes the wall of ecsim +4% (1 rank) and +7% (2 ranks) despite faster pushers. The cause is not investigated; first candidates are the per-cell sort and the access pattern of the moment and momentum loops over the new storage.

Noise: the 2-rank L1-miss values have `sd` comparable to the mean (e.g. eccapfim `0.232 ± 0.308`), do not read them. `perf record` includes diagnostics.

## Tables

### basic

1 rank x 4 threads

| metric | before | after | after/before |
| ------ | ------ | ----- | ------------ |
| stage: Diagnostics run [ms/step] | 66.62 ± 3.45 | 119.40 ± 0.54 | 1.792 |
| stage: Main Stage [ms/step] | 343.51 ± 8.68 | 245.26 ± 1.50 | 0.714 |
| event: update_cells [ms/step] | 18.92 ± 2.49 | 8.41 ± 0.27 | 0.445 |
| IPC | 1.5764 ± 0.0277 | 2.0778 ± 0.0064 | 1.318 |
| final wK | 1.2025 | 1.2025 | 1.000 |
| wall [ms/step] | 410.15 ± 12.13 | 364.67 ± 2.02 | 0.889 |
| perf: L1-dcache-load-misses [G/step] | 0.017 ± 0.002 | 0.008 ± 0.001 | 0.459 |
| perf: cycles [G/step] | 6.565 ± 0.117 | 5.971 ± 0.019 | 0.910 |
| perf: instructions [G/step] | 10.347 ± 0.001 | 12.406 ± 0.001 | 1.199 |

2 ranks x 2 threads

| metric | before | after | after/before |
| ------ | ------ | ----- | ------------ |
| stage: Diagnostics run [ms/step] | 181.38 ± 0.54 | 310.23 ± 4.84 | 1.710 |
| stage: Main Stage [ms/step] | 650.77 ± 10.81 | 556.65 ± 9.32 | 0.855 |
| event: update_cells [ms/step] | 26.37 ± 0.60 | 19.14 ± 0.67 | 0.726 |
| IPC | 2.3561 ± 0.0100 | 2.4316 ± 0.0148 | 1.032 |
| wall [ms/step] | 832.33 ± 11.09 | 867.00 ± 12.17 | 1.042 |
| perf: L1-dcache-load-misses [G/step] | 0.047 ± 0.063 | 0.010 ± 0.004 | 0.219 |
| perf: cycles [G/step] | 16.753 ± 0.629 | 17.324 ± 0.305 | 1.034 |
| perf: instructions [G/step] | 39.469 ± 1.399 | 42.126 ± 0.852 | 1.067 |

`perf record`, 1 rank, self time

| before % | before symbol | after % | after symbol |
| -------- | ------------- | ------- | ------------ |
| 29.11 | `EsirkepovDecomposition::process` | 21.48 | `spline_of_2nd_order` |
| 14.74 | `spline_of_2nd_order` | 17.63 | `Shape::fill` |
| 14.69 | `interfaces::Particles::calculate_moments` | 15.95 | `interfaces::Particles::calculate_moments` |
| 12.88 | `Shape::fill` | 11.52 | `MomentumConservation::calculate` |
| 6.61 | `MomentumConservation::calculate` | 11.02 | `EsirkepovDecomposition::process` |
| 5.17 | `SimpleInterpolation::process` | 5.76 | `SimpleInterpolation::process` |

### ecsim

1 rank x 4 threads

| metric | before | after | after/before |
| ------ | ------ | ----- | ------------ |
| stage: Advance field [ms/step] | 162.86 ± 4.77 | 158.94 ± 1.04 | 0.976 |
| stage: Diagnostics run [ms/step] | 64.92 ± 1.06 | 117.75 ± 0.13 | 1.814 |
| stage: First push [ms/step] | 91.30 ± 1.97 | 75.10 ± 0.36 | 0.823 |
| stage: Second push [ms/step] | 23.99 ± 0.92 | 13.35 ± 0.16 | 0.557 |
| stage: Main Stage [ms/step] | 85.73 ± 0.70 | 79.55 ± 0.25 | 0.928 |
| event: fill_ecsim_current [ms/step] | 34.34 ± 0.28 | 29.30 ± 0.11 | 0.853 |
| event: first_push [ms/step] | 3.84 ± 0.28 | 0.89 ± 0.03 | 0.232 |
| event: update_cells [ms/step] | 32.18 ± 1.80 | 16.31 ± 0.16 | 0.507 |
| event: update_cells_with_assembly [ms/step] | 32.30 ± 1.80 | 16.43 ± 0.16 | 0.509 |
| IPC | 1.5376 ± 0.0104 | 1.7511 ± 0.0039 | 1.139 |
| wall [ms/step] | 429.62 ± 2.15 | 445.52 ± 1.37 | 1.037 |
| perf: L1-dcache-load-misses [G/step] | 0.039 ± 0.000 | 0.038 ± 0.001 | 0.981 |
| perf: cycles [G/step] | 4.380 ± 0.030 | 5.051 ± 0.012 | 1.153 |
| perf: instructions [G/step] | 6.734 ± 0.001 | 8.844 ± 0.001 | 1.313 |

2 ranks x 2 threads

| metric | before | after | after/before |
| ------ | ------ | ----- | ------------ |
| stage: Advance field [ms/step] | 311.47 ± 4.59 | 326.58 ± 1.71 | 1.049 |
| stage: Diagnostics run [ms/step] | 180.76 ± 0.71 | 312.72 ± 2.35 | 1.730 |
| stage: First push [ms/step] | 175.79 ± 2.61 | 136.44 ± 1.77 | 0.776 |
| stage: Second push [ms/step] | 65.12 ± 1.89 | 38.13 ± 0.31 | 0.585 |
| event: fill_ecsim_current [ms/step] | 71.95 ± 0.55 | 62.49 ± 0.18 | 0.868 |
| event: first_push [ms/step] | 8.22 ± 1.04 | 1.17 ± 0.07 | 0.143 |
| event: update_cells [ms/step] | 45.27 ± 2.07 | 34.09 ± 1.03 | 0.753 |
| event: update_cells_with_assembly [ms/step] | 83.97 ± 2.20 | 39.74 ± 1.09 | 0.473 |
| IPC | 2.5358 ± 0.0189 | 2.5440 ± 0.0056 | 1.003 |
| wall [ms/step] | 955.00 ± 10.90 | 1026.00 ± 5.00 | 1.074 |
| perf: cycles [G/step] | 37.677 ± 0.317 | 38.390 ± 0.367 | 1.019 |
| perf: instructions [G/step] | 95.538 ± 0.640 | 97.662 ± 0.720 | 1.022 |

`perf record`, 1 rank, self time

| before % | before symbol | after % | after symbol |
| -------- | ------------- | ------- | ------------ |
| 21.74 | `interfaces::Particles::calculate_moments` | 19.4 | `interfaces::Particles::calculate_moments` |
| 11.96 | `ecsim::Particles::decompose_ecsim_current` | 13.84 | `spline_of_2nd_order` |
| 10.91 | `MatLUFactorNumeric_SeqAIJ` | 13.37 | `MomentumConservation::calculate` |
| 9.57 | `MomentumConservation::calculate` | 11.07 | `Shape::fill` |
| 8.13 | `spline_of_2nd_order` | 9.48 | `MatLUFactorNumeric_SeqAIJ` |
| 6.07 | `Shape::fill` | 9.1 | `ecsim::Particles::decompose_ecsim_current` |

### ecsimcorr

1 rank x 4 threads

| metric | before | after | after/before |
| ------ | ------ | ----- | ------------ |
| stage: Advance field [ms/step] | 156.70 ± 6.06 | 164.88 ± 2.16 | 1.052 |
| stage: Clear sources [ms/step] | 4.65 ± 0.48 | 0.95 ± 0.04 | 0.204 |
| stage: Correct fields [ms/step] | 8.86 ± 0.04 | 9.13 ± 0.11 | 1.031 |
| stage: Diagnostics run [ms/step] | 65.65 ± 1.76 | 120.03 ± 1.65 | 1.828 |
| stage: Final update [ms/step] | 10.19 ± 0.30 | 2.02 ± 0.08 | 0.198 |
| stage: First push [ms/step] | 267.09 ± 3.30 | 170.35 ± 1.12 | 0.638 |
| stage: Second push [ms/step] | 200.46 ± 2.21 | 119.56 ± 13.40 | 0.596 |
| event: fill_ecsim_current [ms/step] | 34.35 ± 0.42 | 29.78 ± 0.47 | 0.867 |
| event: final_update [ms/step] | 8.41 ± 0.29 | 0.33 ± 0.03 | 0.039 |
| event: first_push [ms/step] | 177.95 ± 1.19 | 95.08 ± 0.25 | 0.534 |
| event: second_push [ms/step] | 182.34 ± 0.59 | 110.50 ± 13.13 | 0.606 |
| event: update_cells [ms/step] | 36.21 ± 3.46 | 17.40 ± 0.89 | 0.481 |
| IPC | 1.3486 ± 0.0072 | 1.9102 ± 0.0511 | 1.416 |
| wall [ms/step] | 800.17 ± 3.06 | 667.50 ± 14.03 | 0.834 |
| perf: L1-dcache-load-misses [G/step] | 0.061 ± 0.000 | 0.046 ± 0.002 | 0.752 |
| perf: cycles [G/step] | 10.553 ± 0.056 | 8.562 ± 0.232 | 0.811 |
| perf: instructions [G/step] | 14.231 ± 0.001 | 16.347 ± 0.001 | 1.149 |

2 ranks x 2 threads

| metric | before | after | after/before |
| ------ | ------ | ----- | ------------ |
| stage: Advance field [ms/step] | 317.51 ± 2.73 | 322.61 ± 4.35 | 1.016 |
| stage: Clear sources [ms/step] | 13.59 ± 0.73 | 5.77 ± 0.18 | 0.424 |
| stage: Correct fields [ms/step] | 330.13 ± 2.21 | 320.15 ± 18.70 | 0.970 |
| stage: Diagnostics run [ms/step] | 181.89 ± 1.56 | 309.32 ± 4.05 | 1.701 |
| stage: Final update [ms/step] | 38.80 ± 0.56 | 21.87 ± 0.70 | 0.564 |
| stage: First push [ms/step] | 421.05 ± 7.16 | 320.88 ± 3.44 | 0.762 |
| stage: Second push [ms/step] | 307.27 ± 10.61 | 238.32 ± 7.34 | 0.776 |
| event: final_update [ms/step] | 20.38 ± 0.18 | 3.72 ± 0.38 | 0.182 |
| event: first_push [ms/step] | 250.57 ± 3.58 | 184.57 ± 3.18 | 0.737 |
| event: second_push [ms/step] | 244.13 ± 8.88 | 200.79 ± 7.30 | 0.822 |
| event: update_cells [ms/step] | 54.38 ± 4.88 | 37.88 ± 2.57 | 0.697 |
| event: update_cells_with_assembly [ms/step] | 90.70 ± 4.76 | 42.57 ± 1.54 | 0.469 |
| IPC | 2.4696 ± 0.0039 | 2.5337 ± 0.0073 | 1.026 |
| wall [ms/step] | 1831.17 ± 27.10 | 1751.50 ± 32.67 | 0.956 |
| perf: cycles [G/step] | 47.129 ± 0.529 | 46.684 ± 0.646 | 0.991 |
| perf: instructions [G/step] | 116.390 ± 1.335 | 118.279 ± 1.498 | 1.016 |

`perf record`, 1 rank, self time

| before % | before symbol | after % | after symbol |
| -------- | ------------- | ------- | ------------ |
| 38.78 | `EsirkepovDecomposition::process` | 15.79 | `EsirkepovDecomposition::process` |
| 9.0 | `interfaces::Particles::calculate_moments` | 14.74 | `spline_of_2nd_order` |
| 8.53 | `spline_of_2nd_order` | 13.11 | `Shape::fill` |
| 8.4 | `Shape::fill` | 11.5 | `interfaces::Particles::calculate_moments` |
| 4.91 | `ecsim::Particles::decompose_ecsim_current` | 8.66 | `MomentumConservation::calculate` |
| 4.22 | `MatLUFactorNumeric_SeqAIJ` | 5.45 | `ecsim::Particles::decompose_ecsim_current` |

### eccapfim

1 rank x 4 threads

| metric | before | after | after/before |
| ------ | ------ | ----- | ------------ |
| stage: Calc iteration [ms/step] | 3052.48 ± 21.22 | 3095.69 ± 45.65 | 1.014 |
| stage: Diagnostics run [ms/step] | 12.94 ± 0.25 | 25.61 ± 0.26 | 1.979 |
| stage: After iteration [ms/step] | 2.15 ± 0.42 | 2.03 ± 0.02 | 0.945 |
| stage: Init iteration [ms/step] | 1.52 ± 0.23 | 0.16 ± 0.04 | 0.103 |
| event: form_iteration [ms/step] | - | 3080.47 ± 44.32 | - |
| event: prepare_storage [ms/step] | - | 0.11 ± 0.04 | - |
| event: update_cells [ms/step] | 2.09 ± 0.42 | 1.98 ± 0.02 | 0.947 |
| IPC | 2.0933 ± 0.0114 | 2.1188 ± 0.0192 | 1.012 |
| wall [ms/step] | 3115.56 ± 20.16 | 3169.56 ± 47.13 | 1.017 |
| perf: L1-dcache-load-misses [G/step] | 0.055 ± 0.002 | 0.058 ± 0.003 | 1.050 |
| perf: cycles [G/step] | 51.274 ± 0.278 | 52.027 ± 0.478 | 1.015 |
| perf: instructions [G/step] | 107.331 ± 0.005 | 110.230 ± 0.012 | 1.027 |

2 ranks x 2 threads

| metric | before | after | after/before |
| ------ | ------ | ----- | ------------ |
| stage: Calc iteration [ms/step] | 7613.00 ± 75.67 | 7668.33 ± 83.80 | 1.007 |
| stage: Diagnostics run [ms/step] | 79.98 ± 1.79 | 117.05 ± 4.48 | 1.463 |
| stage: After iteration [ms/step] | 32.97 ± 1.63 | 17.10 ± 0.42 | 0.519 |
| event: form_iteration [ms/step] | - | 6387.00 ± 32.33 | - |
| event: update_cells [ms/step] | 10.24 ± 0.95 | 10.86 ± 0.79 | 1.060 |
| IPC | 2.5338 ± 0.0106 | 2.5236 ± 0.0161 | 0.996 |
| wall [ms/step] | 8014.44 ± 76.69 | 8078.89 ± 100.79 | 1.008 |
| perf: cycles [G/step] | 212.119 ± 12.767 | 209.923 ± 17.010 | 0.990 |
| perf: instructions [G/step] | 537.510 ± 33.682 | 529.940 ± 46.436 | 0.986 |

`perf record`, 1 rank, self time

| before % | before symbol | after % | after symbol |
| -------- | ------------- | ------- | ------------ |
| 18.97 | `spline_of_2nd_order` | 18.33 | `SimpleInterpolation::process` |
| 18.61 | `SimpleInterpolation::process` | 18.19 | `spline_of_2nd_order` |
| 15.72 | `ImplicitEsirkepov::Shape::setup` | 15.84 | `ImplicitEsirkepov::Shape::setup` |
| 14.31 | `Shape::fill` | 14.68 | `Shape::fill` |
| 6.78 | `eccapfim::Particles::form_iteration` | 7.2 | `ImplicitEsirkepov::interpolate` |
| 6.69 | `ImplicitEsirkepov::interpolate` | 5.35 | `ImplicitEsirkepov::decompose` |

Tables are trimmed to the rows that carry information (all rows, with `final wE`, the `Commands run` / `Initialization` / `Clear sources` noise stages and the full top-12 symbol lists, are in `output/results.md`). Per-step values of `stage` and `event` rows are PETSc times divided by the number of steps.
