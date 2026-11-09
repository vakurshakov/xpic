#ifndef SRC_UTILS_PROFILER_H
#define SRC_UTILS_PROFILER_H

#include "src/pch.h"

/**
 * @brief Process-wide timing on top of the PETSc log, reported by `-log_view`
 * and by the `LogView` diagnostic.
 *
 * A Stage is a broad part of a step, stages do not overlap. An Event is a named
 * sub-step, timed inside whichever stage is active when it runs. A Phase is an
 * event that runs inside a threaded region. Register once, then push/pop by
 * the returned handle, e.g.:
 *
 * @code
 * PetscInt solve, push;
 * PetscCall(Profiler::register_event("solve", &solve));
 * PetscCall(Profiler::push_event(solve));
 * ...
 * PetscCall(Profiler::pop_event(solve));
 *
 * PetscCall(Profiler::register_event("push", &push));
 * #pragma omp parallel
 * {
 *   (void)Profiler::push_phase(push);
 *   ...
 *   (void)Profiler::pop_phase(push);
 * }
 * PetscCall(Profiler::flush());
 * @endcode
 *
 * Registering a name twice returns the same handle. Names are shared between
 * implementations: `interpolate`, `push`, `decompose`. Pushes and pops must be
 * properly nested, otherwise the run aborts. Nothing is recorded unless
 * `-log_view` is passed or `enable()` is called (done by the `LogView`).
 *
 * Stages and `push_event` are serial only. `push_phase` is valid both inside
 * and outside parallel regions: outside it is the same as `push_event`, inside
 * it writes to the thread's own slot without PETSc calls, and `flush()` called
 * after the region adds the team-mean time and total call count to the event.
 */
class Profiler {
public:
  static void enable();

  static PetscErrorCode register_stage(const char* name, PetscInt* id);
  static PetscErrorCode push_stage(PetscInt id);
  static PetscErrorCode pop_stage(PetscInt id);

  static PetscErrorCode register_event(const char* name, PetscInt* id);
  static PetscErrorCode push_event(PetscInt id);
  static PetscErrorCode pop_event(PetscInt id);

  static PetscErrorCode push_phase(PetscInt id);
  static PetscErrorCode pop_phase(PetscInt id);

  static PetscErrorCode flush();
};

#endif  // SRC_UTILS_PROFILER_H
