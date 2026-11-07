#include "src/utils/profiler.h"

#include <petsclog.h>
#include <petsctime.h>

#include "tests/common.h"

static constexpr char help[] = "Testing the serial part of the Profiler\n.";

static PetscErrorCode get_event_info(PetscLogEvent event, PetscEventPerfInfo& info)
{
  PetscFunctionBeginUser;
  PetscLogHandler handler;
  PetscCall(PetscLogGetDefaultHandler(&handler));

  PetscLogState state;
  PetscCall(PetscLogHandlerGetState(handler, &state));

  PetscLogStage stage;
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));

  PetscEventPerfInfo* ptr;
  PetscCall(PetscLogHandlerGetEventPerfInfo(handler, stage, event, &ptr));
  info = *ptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void sleep_for(PetscReal seconds)
{
  PetscLogDouble start, now;
  PetscCallAbort(PETSC_COMM_WORLD, PetscTime(&start));
  do {
    PetscCallAbort(PETSC_COMM_WORLD, PetscTime(&now));
  } while (now - start < seconds);
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(PetscLogDefaultBegin());

  PetscInt stage, event, again;
  PetscCall(Profiler::register_stage("test_stage", &stage));
  PetscCall(Profiler::register_event("test_event", &event));

  PetscCall(Profiler::register_stage("test_stage", &again));
  PetscCheck(stage == again, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Same stage name must give the same handle");

  PetscCall(Profiler::register_event("test_event", &again));
  PetscCheck(event == again, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Same event name must give the same handle");
  PetscCheck(stage != event, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Handles must be distinct");

  // Disabled: nothing is recorded
  PetscLogEvent petsc_event;
  PetscCall(PetscLogEventGetId("test_event", &petsc_event));

  PetscCall(Profiler::push(event));
  sleep_for(1e-3);
  PetscCall(Profiler::pop(event));

  PetscEventPerfInfo info;
  PetscCall(get_event_info(petsc_event, info));
  PetscCheck(info.count == 0 && info.time == 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
    "A disabled profiler must record nothing");

  Profiler::enable();

  // Event inside a stage, called twice
  PetscCall(Profiler::push(stage));
  for (PetscInt i = 0; i < 2; i++) {
    PetscCall(Profiler::push(event));
    sleep_for(1e-3);
    PetscCall(Profiler::pop(event));
  }

  PetscCall(get_event_info(petsc_event, info));
  PetscCheck(info.count == 2 && info.time >= 2e-3, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
    "Event must have 2 calls and at least 2 ms, got %d calls, %g s", (int)info.count, info.time);
  PetscCall(Profiler::pop(stage));

  PetscLogStage petsc_stage;
  PetscCall(PetscLogStageGetId("test_stage", &petsc_stage));
  PetscLogHandler handler;
  PetscCall(PetscLogGetDefaultHandler(&handler));
  PetscEventPerfInfo* stage_info;
  PetscCall(PetscLogHandlerGetStagePerfInfo(handler, petsc_stage, &stage_info));
  PetscCheck(stage_info->count == 1 && stage_info->time >= 2e-3, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
    "Stage must have 1 call and at least 2 ms, got %d calls, %g s", (int)stage_info->count, stage_info->time);

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
