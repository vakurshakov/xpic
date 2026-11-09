#ifndef SRC_DIAGNOSTICS_LOG_VIEW_H
#define SRC_DIAGNOSTICS_LOG_VIEW_H

#include <petsclogtypes.h>
#include <petscviewer.h>

#include "src/pch.h"
#include "src/interfaces/diagnostic.h"

/**
 * @brief Provides the possibility to trace the performance of the application.
 * For that we utilize PETSc logging system, but since the nature of the app is
 * iterative, there are options to briefly log the performance at individual
 * timesteps.
 *
 * @li
 * - `EachTimestep` - Timestep summary, separated on stages:
 *   execution time (sec) and a timestep fraction.
 *
 * - `DiagnosePeriodAvg` - Timings are averaged over `diagnose_period`
 *   and events are compressed to have period time fraction > 0.1%.
 *
 * - `AllTimestepsSummary` - Works as a `-log_view` database option,
 *   but we will dump data each `diagnose_period` to avoid losses.
 *
 * - `EachTimestepEvents` - Timestep block, one column per stage: stage header
 *   with its time (sec), then at most 5 events with the largest step time,
 *   each as a percent of the stage, events under 0.1% of the step are dropped.
 * @li
 *
 * @see `PetscLogHandler`, `PetscLogView()`, `PetscLogViewWarnDebugging()`
 */
class LogView : public interfaces::Diagnostic {
public:
  enum Level {
    EachTimestep,
    DiagnosePeriodAvg,
    AllTimestepsSummary,
    EachTimestepEvents
  };

  LogView(Level level);
  PetscErrorCode finalize() override;
  PetscErrorCode diagnose(PetscInt t) override;

private:
  PetscErrorCode level_0_init();
  PetscErrorCode level_0_impl(PetscInt t);
  PetscErrorCode level_1_impl(PetscInt t);
  PetscErrorCode level_2_impl(PetscInt t);
  PetscErrorCode level_3_impl(PetscInt t);

  PetscErrorCode init();
  PetscErrorCode warn();
  PetscErrorCode pop_stack();
  PetscErrorCode push_stack();

  Level level_;
  PetscLogHandler handler_ = nullptr;
  PetscViewer viewer_ = nullptr;

  PetscIntStack temp_stack_;

  PetscLogDouble prev_time_;
  std::map<std::string, PetscLogDouble> prev_times_;
  std::map<std::pair<PetscInt, PetscInt>, PetscLogDouble> prev_event_times_;
};

#endif  // SRC_DIAGNOSTICS_LOG_VIEW_H
