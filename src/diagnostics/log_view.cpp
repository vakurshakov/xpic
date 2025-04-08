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
  PetscFunctionBeginUser;
  if (t == 0)
    PetscCall(level_0_init());

  PetscCall(pop_stack());
  PetscCall(PetscViewerASCIIPrintf(viewer_, "%5d     ", t));

  // Collected and printed data
  PetscLogDouble loc_time, delta_time, stage_time, frac_time;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)viewer_, &comm));

  PetscMPIInt size;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscTime(&loc_time));
  delta_time = (loc_time - prev_time_);
  prev_time_ = loc_time;

  PetscLogDouble max, tot;
  PetscCallMPI(MPIU_Allreduce(&delta_time, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCallMPI(MPIU_Allreduce(&delta_time, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  PetscCall(PetscViewerASCIIPrintf(viewer_, "%9.3e    ", max));

  PetscLogState state;
  PetscInt num_stages;
  PetscCall(PetscLogHandlerGetState(handler_, &state));
  PetscCall(PetscLogStateGetNumStages(state, &num_stages));

  for (PetscLogStage stage = 0; stage < num_stages; ++stage) {
    const char* name;
    PetscEventPerfInfo* info;

    PetscCall(PetscLogStageGetName(stage, &name));
    PetscCall(PetscLogHandlerGetStagePerfInfo(handler_, stage, &info));
    PetscCallMPI(MPIU_Allreduce(&info->time, &stage_time, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));

    delta_time = (stage_time - prev_times_.at(name));
    frac_time = tot > 0.0 ? delta_time / tot : 0.0;
    prev_times_.at(name) = stage_time;
    PetscCall(PetscViewerASCIIPrintf(viewer_, "%-6.4e %5.1f%%  ", delta_time / size, 100.0 * frac_time));
  }

  PetscCall(PetscViewerASCIIPrintf(viewer_, "\n"));
  PetscCall(push_stack());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LogView::level_0_init()
{
  PetscFunctionBeginUser;
  std::string filename = out_dir_ + "/log-EachTimestep.txt";
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename.c_str(), &viewer_));
  PetscCall(warn());

  PetscLogState state;
  PetscInt num_stages;
  PetscCall(PetscLogHandlerGetState(handler_, &state));
  PetscCall(PetscLogStateGetNumStages(state, &num_stages));

  PetscCall(PetscViewerASCIIPrintf(viewer_, "Timestep  Total_[sec]  "));

  for (PetscLogStage stage = 0; stage < num_stages; ++stage) {
    const char* name;
    PetscCall(PetscLogStageGetName(stage, &name));
    prev_times_.insert(std::make_pair(name, 0.0));

    /// @note To simplify the analysis later we replace separate names to be continuous
    std::string fname(name);
    std::replace(fname.begin(), fname.end(), ' ', '_');
    PetscCall(PetscViewerASCIIPrintf(viewer_, "%-19s", fname.c_str()));
  }

  PetscCall(PetscViewerASCIIPrintf(viewer_, "\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LogView::level_1_impl(PetscInt t)
{
  if (t % diagnose_period_ != 0)
    return PETSC_SUCCESS;

  PetscFunctionBeginUser;
  std::string filename = out_dir_ + "/log-DiagnosePeriodAvg.txt";
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename.c_str(), &viewer_));
  PetscCall(pop_stack());
  PetscCall(warn());

  PetscLogDouble loc_time, delta_time, period_time, stage_time;
  PetscLogDouble frac_period_time, frac_stage_time;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)viewer_, &comm));

  PetscMPIInt size;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscTime(&loc_time));
  delta_time = (loc_time - prev_time_);
  prev_time_ = loc_time;

  PetscLogDouble max, tot;
  PetscCallMPI(MPIU_Allreduce(&delta_time, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCallMPI(MPIU_Allreduce(&delta_time, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  period_time = tot;

  PetscCall(PetscViewerASCIIPrintf(viewer_, "Total time (sec):     %5.3e\n", max));
  PetscCall(PetscViewerASCIIPrintf(viewer_, "\nSummary of Stages:    ------------ Time ------------\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer_, "                        Comm-Avg  Period-Avg  %%Total\n"));

  PetscLogState state;
  PetscInt num_stages, num_events;
  PetscCall(PetscLogHandlerGetState(handler_, &state));
  PetscCall(PetscLogStateGetNumStages(state, &num_stages));
  PetscCall(PetscLogStateGetNumEvents(state, &num_events));

  for (PetscLogStage stage = 0; stage < num_stages; ++stage) {
    const char* stage_name;
    PetscEventPerfInfo* stage_info;

    PetscCall(PetscLogStageGetName(stage, &stage_name));
    PetscCall(PetscLogHandlerGetStagePerfInfo(handler_, stage, &stage_info));
    PetscCallMPI(MPIU_Allreduce(&stage_info->time, &stage_time, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));

    if (!prev_times_.contains(stage_name))
      prev_times_.insert(std::make_pair(stage_name, 0.0));

    delta_time = (stage_time - prev_times_.at(stage_name));
    frac_period_time = delta_time / period_time;
    if (100.0 * frac_period_time < 0.1)
      continue;

    PetscCall(PetscViewerASCIIPrintf(viewer_, "%2d:  %15s: %6.4e  %6.4e  %5.1f%%\n", stage, stage_name, delta_time / size, delta_time / diagnose_period, 100.0 * frac_period_time));
  }

  PetscCall(PetscViewerASCIIPrintf(viewer_, "\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer_, "Event                Max         Avg        %%P     %%S\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer_, "------------------------------------------------------\n"));

  for (PetscLogStage stage = 0; stage < num_stages; ++stage) {
    const char* stage_name;
    PetscEventPerfInfo* stage_info;

    bool stage_init = false;
    PetscCall(PetscLogStageGetName(stage, &stage_name));
    PetscCall(PetscLogHandlerGetStagePerfInfo(handler_, stage, &stage_info));

    PetscCallMPI(MPIU_Allreduce(&stage_info->time, &stage_time, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
    PetscLogDouble delta_stage_time = stage_time - prev_times_.at(stage_name);

    if (100.0 * delta_stage_time / period_time < 0.1)
      continue;

    for (PetscLogEvent event = 0; event < num_events; ++event) {
      const char* event_name;
      PetscEventPerfInfo* event_info;

      PetscCall(PetscLogEventGetName(event, &event_name));
      PetscCall(PetscLogHandlerGetEventPerfInfo(handler_, stage, event, &event_info));

      PetscInt maxC;
      PetscCallMPI(MPIU_Allreduce(&event_info->time, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
      PetscCallMPI(MPIU_Allreduce(&event_info->time, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
      PetscCallMPI(MPIU_Allreduce(&event_info->count, &maxC, 1, MPI_INT, MPI_MAX, comm));
      if (maxC == 0)
        continue;

      // clang-format off
      std::string max_name = std::string(stage_name) + std::string(event_name) + "max";
      std::string tot_name = std::string(stage_name) + std::string(event_name) + "tot";
      // clang-format on

      if (!prev_times_.contains(max_name))
        prev_times_.insert(std::make_pair(max_name, 0.0));

      if (!prev_times_.contains(tot_name))
        prev_times_.insert(std::make_pair(tot_name, 0.0));

      delta_time = (tot - prev_times_.at(tot_name));
      frac_period_time = delta_time / period_time;

      if (100.0 * frac_period_time < 0.1)
        continue;

      if (!stage_init) {
        PetscCall(PetscViewerASCIIPrintf(viewer_, "\n--- Event Stage %d: %s\n\n", stage, stage_name));
        stage_init = true;
      }

      frac_stage_time = delta_time / delta_stage_time;
      delta_time = (max - prev_times_.at(max_name));
      PetscCall(PetscViewerASCIIPrintf(viewer_, "%-16s  %5.4e  %5.4e  %5.1f  %5.1f\n", event_name, delta_time, delta_time / diagnose_period, 100.0 * frac_period_time, 100.0 * frac_stage_time));

      prev_times_.at(max_name) = max;
      prev_times_.at(tot_name) = tot;
    }

    prev_times_.at(stage_name) = stage_time;
  }

  PetscCall(push_stack());
  PetscCall(PetscViewerDestroy(&viewer_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LogView::level_2_impl(PetscInt t)
{
  if (t % diagnose_period_ != 0)
    return PETSC_SUCCESS;

  PetscFunctionBeginUser;
  std::string filename = out_dir_ + "/log-AllTimestepsSummary.txt";
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename.c_str(), &viewer_));
  PetscCall(PetscLogView(viewer_));
  PetscCall(PetscViewerDestroy(&viewer_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LogView::init()
{
  PetscFunctionBeginUser;
  PetscCall(PetscLogDefaultBegin());
  prev_time_ = petsc_BaseTime;

  if (level_ == AllTimestepsSummary)
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogGetDefaultHandler(&handler_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LogView::pop_stack()
{
  PetscFunctionBeginUser;
  PetscCall(PetscIntStackCreate(&temp_stack_));

  PetscLogStage stage;
  PetscLogState state;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  while (stage >= 0) {
    PetscCall(PetscLogStagePop());
    PetscCall(PetscIntStackPush(temp_stack_, stage));
    PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LogView::push_stack()
{
  PetscFunctionBeginUser;
  PetscLogStage stage;
  PetscBool is_empty;
  PetscCall(PetscIntStackEmpty(temp_stack_, &is_empty));
  while (!is_empty) {
    PetscCall(PetscIntStackPop(temp_stack_, &stage));
    PetscCall(PetscLogStagePush(stage));
    PetscCall(PetscIntStackEmpty(temp_stack_, &is_empty));
  }
  PetscCall(PetscIntStackDestroy(temp_stack_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LogView::warn()
{
  PetscFunctionBeginUser;
  if (PetscDefined(USE_DEBUG)) {
    PetscCall(PetscViewerASCIIPrintf(viewer_, "\n\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer_, "      ##########################################################\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer_, "      #                                                        #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer_, "      #                       WARNING!!!                       #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer_, "      #                                                        #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer_, "      #   This code was compiled with a debugging option.      #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer_, "      #   To get timing results run ./configure                #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer_, "      #   using --with-debugging=no, the performance will      #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer_, "      #   be generally two or three times faster.              #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer_, "      #                                                        #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer_, "      ##########################################################\n\n\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LogView::finalize()
{
  PetscFunctionBeginUser;
  if (viewer_)
    PetscCall(PetscViewerDestroy(&viewer_));
  PetscFunctionReturn(PETSC_SUCCESS);
}
