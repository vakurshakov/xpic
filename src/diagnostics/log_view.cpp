#include "log_view.h"

#include <petsclog.h>
#include <petsctime.h>

#include "src/utils/configuration.h"

LogView::LogView(Level level)
  : interfaces::Diagnostic(CONFIG().out_dir), level_(level)
{
}

PetscErrorCode LogView::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t == 0)
    PetscCall(init());

  PetscFunctionBegin;
  switch (level_) {
    case EachTimestep:
      PetscCall(level_0_impl(t));
      break;
    case DiagnosePeriodAvg:
      PetscCall(level_1_impl(t));
      break;
    case AllTimestepsSummary:
      PetscCall(level_2_impl(t));
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LogView::level_0_impl(PetscInt t)
{
  PetscCall(PetscViewerASCIIPrintf(viewer_, "%5d     ", t));

  // Collected and printed data
  PetscLogDouble loc_time, delta_time, step_time, stage_time, frac_time;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)viewer_, &comm));

  PetscMPIInt size, rank;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscTime(&loc_time));
  delta_time = (loc_time - prev_loc_time_);
  prev_loc_time_ = loc_time;

  PetscLogDouble max, tot;
  PetscCallMPI(MPIU_Allreduce(&delta_time, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCallMPI(MPIU_Allreduce(&delta_time, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  PetscCall(PetscViewerASCIIPrintf(viewer_, "%9.3e    ", max));
  step_time = tot;

  PetscLogState state;
  PetscInt num_stages;
  PetscCall(PetscLogHandlerGetState(handler_, &state));
  PetscCall(PetscLogStateGetNumStages(state, &num_stages));

  for (PetscLogStage stage = 0; stage < num_stages; ++stage) {
    const char* name;
    PetscEventPerfInfo* stage_info;

    PetscCall(PetscLogStageGetName(stage, &name));
    PetscCall(PetscLogHandlerGetStagePerfInfo(handler_, stage, &stage_info));
    PetscCallMPI(MPIU_Allreduce(&stage_info->time, &stage_time, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));

    delta_time = (stage_time - prev_stage_times_.at(name));
    frac_time = step_time > 0.0 ? delta_time / step_time : 0.0;
    prev_stage_times_.at(name) = stage_time;

    PetscCall(PetscViewerASCIIPrintf(viewer_, "%-6.4e %5.1f%%  ", delta_time / size, 100.0 * frac_time));
  }

  PetscCall(PetscViewerASCIIPrintf(viewer_, "\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode LogView::level_1_impl(PetscInt t)
{
  if (t % diagnose_period != 0)
    return PETSC_SUCCESS;

  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LogView::level_2_impl(PetscInt t)
{
  if (t % diagnose_period != 0)
    return PETSC_SUCCESS;

  PetscFunctionBeginUser;
  std::string filename = out_dir_ + "/log-AllTimestepsSummary.dat";
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename.c_str(), &viewer_));
  PetscCall(PetscLogView(viewer_));
  PetscCall(PetscViewerDestroy(&viewer_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LogView::init()
{
  PetscFunctionBeginUser;
  PetscCall(PetscLogDefaultBegin());

  if (level_ == AllTimestepsSummary)
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogHandlerCreate(PETSC_COMM_WORLD, &handler_));
  PetscCall(PetscLogHandlerSetType(handler_, PETSCLOGHANDLERDEFAULT));
  PetscCall(PetscLogHandlerStart(handler_));

  if (level_ != EachTimestep)
    PetscFunctionReturn(PETSC_SUCCESS);

  prev_loc_time_ = petsc_BaseTime;

  /// @note Removing previous file in case of overwriting it
  std::string filename = out_dir_ + "/log-EachTimestep.dat";
  std::filesystem::remove(filename);
  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer_));
  PetscCall(PetscViewerSetType(viewer_, PETSCVIEWERASCII));
  PetscCall(PetscViewerFileSetMode(viewer_, FILE_MODE_APPEND));
  PetscCall(PetscViewerFileSetName(viewer_, filename.c_str()));

  PetscLogState state;
  PetscInt num_stages;
  PetscCall(PetscLogHandlerGetState(handler_, &state));
  PetscCall(PetscLogStateGetNumStages(state, &num_stages));

  PetscCall(PetscViewerASCIIPrintf(viewer_, "Timestep  Total_[sec]  "));

  for (PetscLogStage stage = 0; stage < num_stages; ++stage) {
    const char* name;
    PetscCall(PetscLogStageGetName(stage, &name));
    prev_stage_times_[name] = 0.0;

    /// @note To simplify the analysis later we replace separate names to be continuous
    std::string fname(name);
    std::replace(fname.begin(), fname.end(), ' ', '_');
    PetscCall(PetscViewerASCIIPrintf(viewer_, "%-19s", fname.c_str()));
  }

  PetscCall(PetscViewerASCIIPrintf(viewer_, "\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

LogView::~LogView()
{
  PetscFunctionBeginUser;
  if (handler_) {
    PetscCallVoid(PetscLogHandlerStop(handler_));
    PetscCallVoid(PetscLogHandlerDestroy(&handler_));
  }

  if (viewer_)
    PetscCallVoid(PetscViewerDestroy(&viewer_));
  PetscFunctionReturnVoid();
}
