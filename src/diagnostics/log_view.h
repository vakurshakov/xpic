#ifndef SRC_DIAGNOSTICS_LOG_VIEW_H
#define SRC_DIAGNOSTICS_LOG_VIEW_H

#include <petsclogtypes.h>
#include <petscviewer.h>

#include "src/pch.h"
#include "src/interfaces/diagnostic.h"

/**
 * @brief Provides the possibility to trace the performance of the application.
 * For that we utilize PETSc logging system, @see `PetscLogHandler`, but since
 * the nature of the application is iterative, there is an option to briefly
 * log the performance of the application at each individual timesteps.
 */
class LogView : public interfaces::Diagnostic {
public:
  enum Level {
    EachTimestep,
    DiagnosePeriodAvg,
    AllTimeSum
  };

  LogView(Level level);
  ~LogView() override;

  PetscErrorCode diagnose(PetscInt t) override;

private:
  PetscErrorCode init();

  Level level_;
  PetscLogHandler handler_ = nullptr;
  PetscViewer viewer_ = nullptr;

  PetscLogDouble prev_loc_time_ = 0.0;
  std::map<const char*, PetscLogDouble> prev_stage_times_;
};

#endif  // SRC_DIAGNOSTICS_LOG_VIEW_H
