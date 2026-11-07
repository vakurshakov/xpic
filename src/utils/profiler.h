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
 * PetscInt solve;
 * PetscCall(Profiler::register_event("solve", &solve));
 * PetscCall(Profiler::push(solve));
 * ...
 * PetscCall(Profiler::pop(solve));
 * @endcode
 *
 * Registering a name twice returns the same handle. Names are shared between
 * implementations: `interpolate`, `push`, `decompose`. Pushes and pops must be
 * properly nested, otherwise the run aborts. Nothing is recorded unless
 * `-log_view` is passed or `enable()` is called (done by the `LogView`).
 */
class Profiler {
public:
  static void enable();

  static PetscErrorCode register_stage(const char* name, PetscInt* id);
  static PetscErrorCode register_event(const char* name, PetscInt* id);

  static PetscErrorCode push(PetscInt id);
  static PetscErrorCode pop(PetscInt id);
};

#endif  // SRC_UTILS_PROFILER_H
