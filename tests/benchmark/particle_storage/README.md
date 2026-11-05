# Particle storage suite

Standalone micro-benchmark of particle storage backends (not a simulation, not in ctest).

- [suite.h](suite.h), [suite.cpp](suite.cpp): the benchmark.
- [run.py](run.py): builds the benchmark (Release), runs all backends on 1/2/4/8 ranks, averages the repeats and writes the tables to `output/particle_storage_suite.md`.

```sh
python3 -I tests/benchmark/particle_storage/run.py 5 20
```

Single run (the defaults are exactly this setup; `a`, `b` are skipped on more than one rank):

```sh
mpiexec -np R build/Release/tests/benchmark/particle_storage_suite.out -n 32 -ppc 16 -steps 20 -vth 0.05 -backends lvabcde
```

With `collide` as the 3rd argument, the script runs the collision variants only and writes `output/particle_storage_collide.md`.

## Problem

`Point` / `PointByField` storage is split across `interfaces::Particles::storage`, `drift_kinetic::Particles::dk_{curr,prev}_storage` and `eccapfim::Particles::previous_storage`. Duplicated: `update_cells_{seq,mpi}`, `correct_coordinates` / `g_bound_periodic`, the `MPI_BYTE` migration (`sizeof(Point)` vs `sizeof(PointByField)`), the prev/curr pairing (matched only by list position), and DK's `storage` -> `dk_curr_storage` staging in `commit_added_particles()`.

## Decisions

1. **Per-sort attribute sets.** A sort registers only what it needs: kinetic `r, p`; DK `r, p, p_parallel, mu_p`. Sorts that need the step-start snapshot also register `*_prev` attributes in the same container (alignment by construction). `prepare_storage()` becomes an attribute copy. The prev attributes do not migrate: particles are only moved by `update_cells()` after the solve.
2. **Cell order** is relaxed and restored by a periodic counting sort (`e`). The remaining per-cell consumers (pushers over `storage[g]`, `DistributionMoment`, `ecsim` `storage[g].empty()`, per-cell `Ng`) are adapted to a sorted view or a resort before them. Cell membership is stale during nonlinear iterations, as today.
3. **Backend: flat DMSwarm attributes with surface-only updates (`e`)**, chosen after the spike. `DMSwarmMigrate` is not usable, our own exchange is used. How collisions are served (grouping by cell on demand) is left open, see "Collisions". Neighbour-only exchange is wanted but not critical.
4. **Raw `DMSwarmGetField` arrays with our own sort/exchange** are the access path if the swarm is used. `DMSwarmSortGetPointsPerCell` is out: it is a `qsort` of all points and a shared `DMGetWorkArray` list, unusable in OpenMP loops. Rules for `DMSwarmGetField`: call it once on the main thread, pass raw pointers into the parallel region; a second `GetField` on an active attribute errors; restore every attribute before anything that reallocates (add/remove point, migrate).
5. **Pushers load a local `Point` / `PointByField`, compute, store back.** The value types stay for now.
6. **One ingest/extract path.** Loaders produce kinetic `Point`s. Each sort has `ingest(points)` (DK interpolates B in a batch and applies `from_particle_position` / `from_guiding_center`, i.e. today's `commit_added_particles`) and `extract(predicate)`. `commit_added_particles()` leaves the base interface and DK's staging in base `storage` goes away.

## Tested versions

Per step: snapshot (`prepare`), Boris-like push from the snapshot (`push`), 2nd-order deposit with atomics (`deposit`), `update_cells` (periodic wrap, ownership change, neighbour exchange over 27 slots, re-ordering; reported as `migrate` + `sort`). Charge and particle count are checked every step in every backend.

Gate: push + deposit within 10% of `l`, `update_cells` not slower, no loss on several ranks.

| id | model |
| -- | ----- |
| l | `vector<list<P>>` (current xpic layout), `prev` as `vector<vector<P>>` |
| v | `vector<vector<P>>`, swap-with-last removal, serial local moves |
| a | DMSwarm, `DMSwarmSortGetPointsPerCell`, `DMSwarmMigrate` |
| b | DMSwarm, own counting sort, `DMSwarmMigrate` |
| c | DMSwarm attributes, own counting sort of all particles each step, own neighbour exchange |
| d | DMSwarm attributes as fixed-capacity per-cell blocks (swap-with-last inside a block, append on arrival), own exchange |
| e | DMSwarm flat attributes, **no per-cell order**: one parallel pass wraps every particle and marks those leaving the rank, only those are processed (swap-with-last removal, arrivals appended), own exchange; `-resort 10` additionally runs the counting sort every 10 steps |

Result: `a`, `b` fail (migration, hot loops); `c`, `d`, `e` and `v` pass; **`e` is chosen**. Setup of the tables below: `-n 32 -ppc 16 -vth 0.05` (524288 particles, ~2% change cell per step), Release, 2 OpenMP threads per rank, 16-core machine.

Values are ms/step, `mean ± standard deviation` over 5 runs of 20 steps (repeats are the outer loop); `update_cells` = `migrate` + `sort`. `a`, `b` lose particles on more than one rank, so they run on 1 rank only. Differences below ~1-2 ms are within the noise of this machine.

**1 rank** (5 runs, 20 steps each)

| id | prepare | push | deposit | migrate | sort | update_cells | total |
| -- | ------- | ---- | ------- | ------- | ---- | ------------ | ----- |
| l | 8.2 ± 0.1 | 10.1 ± 0.0 | 73.5 ± 0.9 | 0.0 ± 0.0 | 14.2 ± 0.1 | 14.2 ± 0.1 | 106.0 ± 0.8 |
| v | 4.4 ± 0.2 | 6.5 ± 0.0 | 67.4 ± 0.3 | 0.0 ± 0.0 | 10.8 ± 0.0 | 10.8 ± 0.0 | 89.1 ± 0.2 |
| a | 3.0 ± 0.2 | 49.3 ± 11.8 | 158.2 ± 36.5 | 14.4 ± 0.3 | 32.6 ± 0.2 | 47.0 ± 0.4 | 257.4 ± 48.1 |
| b | 3.0 ± 0.2 | 3.6 ± 0.0 | 66.7 ± 0.4 | 14.4 ± 0.6 | 10.0 ± 0.6 | 24.4 ± 0.8 | 97.8 ± 0.7 |
| c | 3.1 ± 0.2 | 3.7 ± 0.1 | 66.3 ± 0.5 | 2.4 ± 0.0 | 10.1 ± 0.3 | 12.5 ± 0.3 | 85.6 ± 0.7 |
| d | 3.9 ± 0.1 | 8.6 ± 0.1 | 68.0 ± 0.5 | 6.4 ± 0.1 | 3.6 ± 0.0 | 10.0 ± 0.1 | 90.5 ± 0.7 |
| e | 2.7 ± 0.1 | 3.7 ± 0.0 | 66.6 ± 0.8 | 1.4 ± 0.0 | 0.0 ± 0.0 | 1.4 ± 0.0 | 74.4 ± 0.9 |
| e, resort 10 | 2.7 ± 0.1 | 3.7 ± 0.0 | 67.3 ± 0.7 | 1.4 ± 0.0 | 1.1 ± 0.1 | 2.5 ± 0.1 | 76.2 ± 0.8 |

**2 ranks** (5 runs, 20 steps each)

| id | prepare | push | deposit | migrate | sort | update_cells | total |
| -- | ------- | ---- | ------- | ------- | ---- | ------------ | ----- |
| l | 6.4 ± 0.2 | 7.0 ± 0.2 | 36.6 ± 0.5 | 0.0 ± 0.0 | 7.7 ± 0.6 | 7.7 ± 0.6 | 57.7 ± 1.1 |
| v | 3.3 ± 0.0 | 5.1 ± 0.0 | 34.3 ± 0.3 | 0.0 ± 0.0 | 5.6 ± 0.1 | 5.6 ± 0.1 | 48.4 ± 0.3 |
| c | 3.1 ± 0.6 | 2.7 ± 0.2 | 33.9 ± 0.3 | 2.2 ± 0.3 | 6.8 ± 1.4 | 9.0 ± 1.7 | 48.6 ± 1.3 |
| d | 3.5 ± 0.1 | 5.7 ± 0.1 | 34.9 ± 0.5 | 3.7 ± 0.1 | 2.4 ± 0.1 | 6.1 ± 0.1 | 50.2 ± 0.6 |
| e | 2.7 ± 0.3 | 2.9 ± 0.1 | 33.7 ± 0.2 | 1.0 ± 0.0 | 0.0 ± 0.0 | 1.0 ± 0.0 | 40.3 ± 0.5 |
| e, resort 10 | 2.7 ± 0.1 | 2.8 ± 0.1 | 33.9 ± 0.4 | 1.0 ± 0.0 | 0.7 ± 0.1 | 1.7 ± 0.1 | 41.0 ± 0.4 |

**4 ranks** (5 runs, 20 steps each)

| id | prepare | push | deposit | migrate | sort | update_cells | total |
| -- | ------- | ---- | ------- | ------- | ---- | ------------ | ----- |
| l | 5.3 ± 0.1 | 5.9 ± 0.0 | 18.8 ± 0.1 | 0.0 ± 0.0 | 4.5 ± 0.3 | 4.5 ± 0.3 | 34.5 ± 0.3 |
| v | 3.3 ± 0.1 | 4.3 ± 0.1 | 20.8 ± 7.3 | 0.0 ± 0.0 | 3.4 ± 0.4 | 3.4 ± 0.4 | 31.8 ± 7.7 |
| c | 2.6 ± 0.6 | 2.4 ± 0.5 | 18.3 ± 1.7 | 1.8 ± 0.1 | 7.0 ± 0.3 | 8.8 ± 0.5 | 32.2 ± 1.9 |
| d | 3.6 ± 0.0 | 4.3 ± 0.1 | 17.5 ± 0.2 | 2.5 ± 0.0 | 1.5 ± 0.1 | 4.0 ± 0.1 | 29.4 ± 0.2 |
| e | 2.9 ± 0.1 | 2.8 ± 0.2 | 18.4 ± 1.0 | 0.8 ± 0.0 | 0.0 ± 0.0 | 0.8 ± 0.0 | 24.9 ± 0.9 |
| e, resort 10 | 2.9 ± 0.2 | 2.8 ± 0.1 | 17.3 ± 0.1 | 0.8 ± 0.0 | 0.7 ± 0.1 | 1.6 ± 0.1 | 24.6 ± 0.3 |

**8 ranks** (5 runs, 20 steps each)

| id | prepare | push | deposit | migrate | sort | update_cells | total |
| -- | ------- | ---- | ------- | ------- | ---- | ------------ | ----- |
| l | 4.8 ± 0.1 | 5.3 ± 0.1 | 17.5 ± 0.1 | 0.0 ± 0.0 | 3.3 ± 0.1 | 3.3 ± 0.1 | 30.9 ± 0.2 |
| v | 3.3 ± 0.1 | 4.5 ± 0.1 | 17.2 ± 0.1 | 0.0 ± 0.0 | 2.5 ± 0.1 | 2.5 ± 0.1 | 27.5 ± 0.4 |
| c | 2.6 ± 0.3 | 2.6 ± 0.3 | 17.2 ± 0.2 | 1.8 ± 0.2 | 7.5 ± 0.7 | 9.2 ± 0.7 | 31.7 ± 1.1 |
| d | 4.0 ± 0.0 | 4.9 ± 0.1 | 17.1 ± 0.0 | 2.1 ± 0.0 | 1.1 ± 0.1 | 3.2 ± 0.1 | 29.3 ± 0.2 |
| e | 3.2 ± 0.1 | 3.0 ± 0.1 | 17.6 ± 0.3 | 1.2 ± 0.7 | 0.0 ± 0.0 | 1.2 ± 0.7 | 25.0 ± 1.1 |
| e, resort 10 | 3.1 ± 0.1 | 2.8 ± 0.1 | 17.4 ± 0.2 | 0.9 ± 0.0 | 0.6 ± 0.1 | 1.5 ± 0.1 | 24.9 ± 0.2 |

Reading:
- `e` has the lowest total at every rank count (74 / 40 / 25 / 25 ms for 1 / 2 / 4 / 8 ranks) and an `update_cells` of 0.8-1.4 ms, 2-3x below the next best (`v`, `d`) and ~10x below `c`. It touches only the particles that leave the rank (surface of the rank box) instead of the ~2% that change cell (volume), so on 1 rank there is nothing to move at all.
- `v` beats the current list `l` in every phase and is the simplest layout; `d` matches it in `update_cells` but its push is slower (padded blocks, 2x footprint).
- `c` has the fastest push and prepare of the sorted variants but pays 7-10 ms for the sort of all particles every step.
- `a` (PETSc sort access) is unusable in the hot loops; `a` and `b` also fail on several ranks.
- The list is cheap in `update_cells` not because of `std::list` but because it touches only particles that change cell: `v` does the same and is faster.

### Order decay of `e`

The tables use gyrating particles (`-omega 0.1`: orbit radius below one cell), so the cell order of `e` barely decays. With free streaming (`-omega 0 -steps 200`, 1 rank, single runs) the cost of never re-sorting is visible:

| `e` resort period | deposit | sort (amortised) | total |
| ----------------- | ------- | ---------------- | ----- |
| never | 79.1 | 0 | 89.2 |
| 10 | 66.3 | 1.0 | 76.0 |
| 50 | 67.1 | 0.2 | 76.5 |

(`c`, 1 rank, same run: 86.7; `d`: 89.3.) A resort every ~50 steps keeps the deposit at the sorted level at ~0.2 ms/step. The 4-rank runs of this test were too noisy to quote.

Caveat for `e`: the real code iterates by cell in several places (pushers over `storage[g]`, `DistributionMoment` region skip, `ecsim/simulation.cpp` `storage[g].empty()`, per-cell `Ng`), so an unordered layout needs a sorted view or a resort before those, which this part of the benchmark does not model. Collisions are the first such consumer, modelled below.

### Why the first spike failed the gate

The first spike (`b`, `DMSwarmMigrate`, one rank, 4 threads) had `update_cells` ~7x slower than the list (~15 ms of `DMSwarmMigrate`, which locates every particle, and ~20 ms of serial sort) and push/deposit on par. Both are avoidable with our own exchange and sort (backends `c`-`e`).

### `DMSwarmMigrate` over a DMDA

PETSc `DMLocatePoints_DA_Regular` (`dageometry.c`) locates a point inside the local **node** bounding box and has `TODO: Check for periodicity`.

- Stencil width 0: no ghost nodes, so the last owned cell of every rank but the last has no upper node and is "not found"; `DMSwarmMigrate(.., PETSC_TRUE)` deletes its particles.
- Stencil width >= 1: ghost cells count as found, so a particle that crossed into the neighbour's cell is "located" locally and is never sent.
- `DMSwarmSortGetAccess` on a DMDA needs stencil width >= 1 (`DMDAGetElements`).

Unchanged in PETSc v3.26.0 and `main` (checked 2026-10-27; the local checkout is v3.24.3), so an upgrade does not help. A DMSwarm migration would need a different cell DM (e.g. a DMShell) or none. Backends `c`-`e` use the swarm only as an attribute registry.

### Other findings

- The own exchange (27 slots, count messages then `MPI_BYTE` payloads) is the same code for `l`, `v`, `c`, `d`, `e`. One tag per phase (`-xpic_tags`, as `update_cells_mpi`) is safe: messages between a pair of ranks are non-overtaking and the destination cell is recomputed from the coordinates.
- Preallocation (`-buffer 0 / 5 / 20` %) does not change the migrate time; keep a small buffer for `inject_particles`-style appends.
- `d`: block capacity factor 1.2-3.0 changes totals by < 3%; a full block triggers a rebuild with `2 * max count`.
- Swap-with-last removal does not preserve particle order, so the `prev[g][i]` pairing must be re-snapshotted after `update_cells`, as `prepare_storage()` already does.

## Collisions

Collisions are the per-cell consumer that needs cell-grouped particles, so they decide whether dropping the per-cell order (`e`) is affordable.

Same-species binary collisions inside a cell (Takizuka-Abe: random pairing by a shuffled index array, Coulomb-like angle with `nu_dt = 0.05 (vth/|u|)^3`, odd particle of a cell skipped, per-cell RNG seed so the result does not depend on the thread count). One kernel for all backends, called after `update_cells`; momentum and kinetic energy are checked around every call (|dp| < 1e-8, |dE|/E < 1e-10), and that the pairs did scatter. Not collided here: `l`, `d` (no extra information), `a`, `b` (broken).

| id | model |
| -- | ----- |
| v | collides `storage[g]` in place, no extra pass |
| c | sorted every step already (part of `update_cells`), collisions use the offsets |
| e + sort | cell order of `e` is stale: counting sort (permutes the attributes) before the collisions, in `collide` |
| e + index | counting pass that produces only `perm` (indices by cell), attributes stay in place, the kernel gathers through `perm`; in `collide` |

Options: `-collide k` (every k steps), `-collide_index 1` (`e + index`). `collide` includes the grouping for `e`. Same setup and statistics as above; full tables, also for collisions every 10 steps on 2 and 4 ranks, are in `output/particle_storage_collide.md` after `run.py ... collide`.

**Collisions every step, 1 rank**

| id | prepare | push | deposit | update_cells | collide | total |
| -- | ------- | ---- | ------- | ------------ | ------- | ----- |
| v | 4.6 ± 0.2 | 7.7 ± 0.2 | 70.2 ± 0.4 | 11.1 ± 0.1 | 20.5 ± 0.2 | 114.2 ± 1.0 |
| c | 3.0 ± 0.3 | 4.9 ± 0.1 | 69.0 ± 0.5 | 13.2 ± 0.7 | 17.0 ± 0.3 | 107.1 ± 1.1 |
| e + sort | 3.0 ± 0.1 | 4.6 ± 0.1 | 69.0 ± 0.5 | 1.6 ± 0.0 | 27.2 ± 0.4 | 105.4 ± 0.9 |
| e + index | 2.9 ± 0.2 | 5.5 ± 0.2 | 69.1 ± 0.7 | 1.6 ± 0.0 | 23.6 ± 0.6 | 102.7 ± 0.9 |

**Collisions every step, 8 ranks**

| id | prepare | push | deposit | update_cells | collide | total |
| -- | ------- | ---- | ------- | ------------ | ------- | ----- |
| v | 2.5 ± 0.3 | 3.9 ± 0.3 | 17.5 ± 0.4 | 2.8 ± 0.1 | 5.2 ± 0.3 | 32.0 ± 0.8 |
| c | 1.6 ± 0.2 | 2.0 ± 0.2 | 17.2 ± 0.1 | 7.0 ± 0.4 | 4.1 ± 0.2 | 31.9 ± 0.7 |
| e + sort | 1.8 ± 0.2 | 2.1 ± 0.5 | 17.2 ± 0.2 | 1.3 ± 0.2 | 8.5 ± 0.7 | 30.9 ± 1.4 |
| e + index | 2.0 ± 0.3 | 2.4 ± 0.4 | 17.6 ± 0.3 | 1.2 ± 0.2 | 6.0 ± 0.4 | 29.3 ± 1.1 |

**Collisions every 10 steps, 1 rank / 8 ranks** (totals; `collide` amortised: 1.7-2.8 ms / 0.4-0.8 ms)

| id | 1 rank | 8 ranks |
| -- | ------ | ------- |
| v | 95.6 ± 0.2 | 28.8 ± 0.6 |
| c | 92.1 ± 0.5 | 31.8 ± 1.3 |
| e + sort | 80.8 ± 0.4 | 25.6 ± 0.4 |
| e + index | 80.4 ± 0.4 | 25.2 ± 0.4 |

Reading (differences below ~2 ms are noise; the 4-rank deposit of `e` is noisy again):
- The kernel is expensive: 17-20 ms/step for 524288 particles on 1 rank, 3-4x a push (Box-Muller `log` + `cos`, `sin`/`cos` of phi, 2 `sqrt`), so with collisions every step the total grows by ~20%, and the layout matters much less than the physics cost.
- On the same sorted data the kernel runs 17 ms for `c` (SoA, offsets) against 20.5 ms for `v` (AoS in per-cell vectors): the SoA gain is ~3 ms.
- Grouping `e` costs 6.6 ms for the index (1 rank, `collide` 23.6 vs 17.0 of `c`) and ~10 ms for the physical sort. Both are paid on top of the kernel, but `e` saves 10-12 ms in `update_cells`, so **collisions every step do not turn the ranking over**: `e + index` / `e + sort` are 2-10% below `v` on 1, 2 and 8 ranks (4 ranks: all within the noise).
- With collisions every 10 steps the grouping is amortised and `e` leads by ~15 ms on 1 rank and ~3-6 ms on 8 ranks. `c` is the worst: it pays its per-step sort whether collisions run or not.
- `e + index` is about `e + sort` in total; the index is ~3-4 ms cheaper per collision step but the attributes stay unsorted, so later gathers lose locality. Free streaming (`-omega 0`, 100 steps, 1 rank, collisions every step, 3 runs): `v` 113-114, `e + sort` 105-107, `e + index` 104-111 ms; the spread of `e + index` is the deposit/push of an order that decays (74 vs 69 ms in 2 of 3 runs), which `e + sort` avoids by sorting before each collision step. With the index a periodic physical resort stays necessary.
- Not modelled: weights (the pair scheme needs the weighted-probability variant for unequal weights), inter-species pairs (two index sets per cell, same cost structure), odd-count cells (Takizuka-Abe three-particle treatment), the drift-kinetic operator (pitch-angle scattering of `p_parallel`, `mu_p`), ppc != 16.
- The kernel is the same in all backends (a loop over cells with a `vel(i)` accessor); `e` additionally needs `build_index()` (30 lines) and the decision when to resort.

**Open:** whether collisions group by `e + sort` or `e + index` (or a resort period shared with the other per-cell consumers) is decided when collisions are ported; both fit `e`.
