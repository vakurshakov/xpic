# eccapfim: list storage vs flat swarm fields (diff 2b)

[suite.cpp](suite.cpp): warm periodic electrons as in `eccapfim_ex4`, `-n 12 -ppc 32` (55296 particles), 3 steps, Anderson SNES with `atol = rtol = 1e-7` (~33 `form_iteration` calls per step), Release, 4 threads in total (1 rank x 4, 2 x 2, 4 x 1). Reproduce with `python3 -I tests/benchmark/eccapfim_warm_plasma/run.py <label> 3 3` (3 repeats; repeats are the outer loop). Values are ms/step, `mean ± standard deviation`.

| phase | meaning |
| ----- | ------- |
| prepare | `prepare_storage()`: snapshot of the step start (`previous_storage` / `save_previous()`) |
| interpolate | `set_fields()` of every Crank-Nicolson iteration, with `cell_traversal()` |
| cn_push | the update of `vh`, `pn.r`, `pn.p` of a Crank-Nicolson iteration, without `set_fields()` |
| decompose | the final `cell_traversal()` and `ImplicitEsirkepov::decompose()` |
| other | the rest of the kernel: `dtau`, residues, bounds, load/store (and the cost of the timers) |
| form_iteration | the whole `Particles::form_iteration()` |
| field solve | SNES time without `form_iteration` |
| update_cells | `update_cells()` after the solve |

_interpolate_ / _cn_push_ / _decompose_ are per-thread times (summed over the threads, divided by their number), measured with `omp_get_wtime()` around the code, so they include ~4% of timer overhead (`perf`: `__vdso_clock_gettime`); _other_ is what remains of `form_iteration`.

**list** (`std::vector<std::vector<Point>> previous_storage`, parallel over cells, `schedule(static)`; diff 2a)

| ranks | prepare | interpolate | cn_push | decompose | other | form_iteration | field solve | update_cells | total |
| ----- | ------- | ----------- | ------- | --------- | ----- | -------------- | ----------- | ------------ | ----- |
| 1 | 0.8 ± 0.0 | 2525.0 ± 14.3 | 37.9 ± 0.1 | 383.3 ± 8.6 | 225.8 ± 30.3 | 3171.9 ± 40.0 | 15.1 ± 0.8 | 2.2 ± 0.1 | 3219.1 ± 41.2 |
| 2 | 0.5 ± 0.0 | 2528.2 ± 1.5 | 38.1 ± 0.1 | 372.2 ± 6.4 | 221.2 ± 12.3 | 3124.6 ± 1.1 | 17.9 ± 0.2 | 1.6 ± 0.1 | 3173.4 ± 1.5 |
| 4 | 0.3 ± 0.0 | 2563.7 ± 5.6 | 38.6 ± 0.1 | 352.0 ± 0.4 | 229.6 ± 5.2 | 3114.9 ± 14.4 | 19.6 ± 1.0 | 1.1 ± 0.0 | 3164.2 ± 15.3 |

**e, `schedule(dynamic, 16)`** over the particles (first attempt)

| ranks | prepare | interpolate | cn_push | decompose | other | form_iteration | field solve | update_cells | total |
| ----- | ------- | ----------- | ------- | --------- | ----- | -------------- | ----------- | ------------ | ----- |
| 1 | 0.1 ± 0.0 | 2537.9 ± 12.5 | 38.0 ± 0.1 | 659.0 ± 5.7 | 152.1 ± 2.3 | 3387.0 ± 18.1 | 16.0 ± 0.5 | 1.9 ± 0.1 | 3434.3 ± 18.2 |
| 2 | 0.1 ± 0.0 | 2612.2 ± 58.2 | 38.6 ± 0.2 | 493.5 ± 65.1 | 280.7 ± 49.4 | 3308.4 ± 12.5 | 18.9 ± 0.2 | 1.6 ± 0.1 | 3358.5 ± 13.8 |
| 4 | 0.1 ± 0.0 | 2570.8 ± 18.4 | 38.8 ± 0.3 | 354.3 ± 2.3 | 256.9 ± 19.2 | 3134.8 ± 35.0 | 19.7 ± 0.3 | 1.1 ± 0.0 | 3184.0 ± 34.8 |

**e, `schedule(static)`** over the particles (final)

| ranks | prepare | interpolate | cn_push | decompose | other | form_iteration | field solve | update_cells | total |
| ----- | ------- | ----------- | ------- | --------- | ----- | -------------- | ----------- | ------------ | ----- |
| 1 | 0.1 ± 0.0 | 2496.2 ± 4.4 | 37.9 ± 0.1 | 374.5 ± 6.3 | 186.9 ± 6.5 | 3095.5 ± 8.0 | 15.1 ± 1.0 | 2.0 ± 0.1 | 3141.2 ± 9.7 |
| 2 | 0.1 ± 0.0 | 2521.5 ± 17.6 | 38.6 ± 0.7 | 377.8 ± 2.5 | 227.0 ± 24.1 | 3110.4 ± 27.5 | 18.1 ± 1.0 | 1.7 ± 0.0 | 3159.4 ± 28.0 |
| 4 | 0.1 ± 0.0 | 2544.4 ± 5.9 | 38.5 ± 0.1 | 349.7 ± 1.3 | 225.0 ± 11.0 | 3085.9 ± 4.4 | 18.4 ± 1.2 | 1.2 ± 0.0 | 3133.3 ± 4.1 |

Reading:
- The step is the kernel: `form_iteration` is 98% of the total. The field solve (~15-20 ms) and `update_cells` (1-2 ms) are negligible next to it.
- _Interpolation_ is 80% of the kernel, _decomposition_ ~12%, the _Crank-Nicolson_ update itself only ~1.2% (38 ms). `perf` agrees: `SimpleInterpolation::process`, `spline_of_2nd_order`, `ImplicitEsirkepov::Shape::setup` and `Shape::fill` are 65% of the samples, `form_iteration` itself 3%. The cost is the shape-function evaluation in `set_fields()`, not the data layout.
- `e` with `static` is level with the list (total 3141 / 3159 / 3133 vs 3219 / 3173 / 3164 ms, within 0-2.5%), _prepare_ drops 3-8x (0.8 -> 0.1 ms), _update_cells_ is the same. The push does not lose to the list, so there is no reason to fall back to `v`.
- `dynamic, 16` is wrong here: threads take neighbouring chunks of the cell-sorted arrays, so the `omp atomic` of _decompose_ hits the same `J` cache lines (decompose 383 -> 659 ms on 1 rank). `static` gives every thread its own contiguous block of cells, as the list did.
- The `perf` profile of the list and of `e` is the same (shares differ by < 1%).
