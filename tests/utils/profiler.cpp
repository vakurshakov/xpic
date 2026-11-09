#include "src/utils/profiler.h"

#include <petsclog.h>

#include "tests/common.h"

static constexpr char help[] =
  "Testing the Profiler: serial use and OMP phases\n.";

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

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(PetscLogDefaultBegin());

  omp_set_num_threads(4);  // slots are sized at registration

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

  PetscCall(Profiler::push_event(event));
  PetscCall(PetscSleep(1e-3));
  PetscCall(Profiler::pop_event(event));

  PetscEventPerfInfo info;
  PetscCall(get_event_info(petsc_event, info));
  PetscCheck(info.count == 0 && info.time == 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
    "A disabled profiler must record nothing");

  Profiler::enable();

  // Event inside a stage, called twice
  PetscCall(Profiler::push_stage(stage));
  for (PetscInt i = 0; i < 2; i++) {
    PetscCall(Profiler::push_event(event));
    PetscCall(PetscSleep(1e-3));
    PetscCall(Profiler::pop_event(event));
  }

  PetscCall(get_event_info(petsc_event, info));
  PetscCheck(info.count == 2 && info.time >= 2e-3, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
    "Event must have 2 calls and at least 2 ms, got %d calls, %g s", info.count, info.time);

  PetscCall(Profiler::pop_stage(stage));

  PetscLogStage petsc_stage;
  PetscCall(PetscLogStageGetId("test_stage", &petsc_stage));
  PetscLogHandler handler;
  PetscCall(PetscLogGetDefaultHandler(&handler));
  PetscEventPerfInfo* stage_info;
  PetscCall(PetscLogHandlerGetStagePerfInfo(handler, petsc_stage, &stage_info));

  PetscCheck(stage_info->count == 1 && stage_info->time >= 2e-3, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
    "Stage must have 1 call and at least 2 ms, got %d calls, %g s", stage_info->count, stage_info->time);

  // Phases inside a team of 4: the mean is over the team, the count is total
  PetscInt phase, partial, mixed, serial;
  PetscCall(Profiler::register_event("test_phase", &phase));
  PetscCall(Profiler::register_event("test_partial", &partial));
  PetscCall(Profiler::register_event("test_mixed", &mixed));
  PetscCall(Profiler::register_event("test_serial", &serial));

#pragma omp parallel num_threads(4)
  {
    (void)Profiler::push_phase(phase);
    (void)PetscSleep(2e-3);
    (void)Profiler::pop_phase(phase);
  }

  PetscEventPerfInfo before;
  PetscCall(get_event_info(phase, before));

  PetscCheck(before.count == 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
    "Threaded phase must not touch PETSc before flush()");

  PetscCall(Profiler::flush());
  PetscCall(get_event_info(phase, info));

  PetscCheck(info.count == 4 && info.time >= 2e-3, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
    "Phase must have 4 calls and mean of about 2 ms, got %d calls, %g s", info.count, info.time);

  // Only thread 0 works: idle threads lower the mean
#pragma omp parallel num_threads(4)
  {
    if (omp_get_thread_num() == 0) {
      (void)Profiler::push_phase(partial);
      (void)PetscSleep(4e-3);
      (void)Profiler::pop_phase(partial);
    }
  }
  PetscCall(Profiler::flush());
  PetscCall(get_event_info(partial, info));

  PetscCheck(info.count == 1 && info.time >= 1e-3, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
    "Partial phase must have 1 call and a mean below the busy thread's time, got %d calls, %g s", info.count, info.time);

  // Serial and threaded use of one event are both counted once
  PetscCall(Profiler::push_phase(mixed));
  PetscCall(PetscSleep(1e-3));
  PetscCall(Profiler::pop_phase(mixed));
#pragma omp parallel num_threads(4)
  {
    (void)Profiler::push_phase(mixed);
    (void)PetscSleep(1e-3);
    (void)Profiler::pop_phase(mixed);
  }
  PetscCall(Profiler::flush());
  PetscCall(Profiler::flush());
  PetscCall(get_event_info(mixed, info));

  PetscCheck(info.count == 5 && info.time >= 2e-3, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
    "Mixed event must have 5 calls and about 2 ms, got %d calls, %g s", info.count, info.time);

  // flush() without threaded use changes nothing
  PetscCall(Profiler::push_event(serial));
  PetscCall(PetscSleep(1e-3));
  PetscCall(Profiler::pop_event(serial));
  PetscCall(get_event_info(serial, before));
  PetscCall(Profiler::flush());
  PetscCall(get_event_info(serial, info));

  PetscCheck(info.count == before.count && info.time == before.time, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
    "flush() must not change serially used events");

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
