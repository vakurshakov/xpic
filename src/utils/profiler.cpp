#include "profiler.h"

#include <stack>

#include <petsclog.h>
#include <petsctime.h>
#include "petscsys.h"

namespace {

// Handles are the PETSc ids, PETSc is the source of truth for names.
struct alignas(64) ThreadSlot {
  struct Acc {
    PetscLogDouble time = 0;
    PetscLogDouble start = 0;
    PetscInt count = 0;
    PetscInt team_size = 0;
  };

  std::vector<Acc> events;      // indexed by event handle
  std::stack<PetscInt> active;  // handles of the pushed phases
};

bool enabled = false;
PetscClassId classid = 0;
std::stack<PetscInt> active_stages;
std::stack<PetscInt> active_events;
std::vector<ThreadSlot> slots;  // indexed by thread number

PetscErrorCode check_option()
{
  PetscFunctionBeginUser;
  PetscBool flg;
  PetscCall(PetscOptionsHasName(nullptr, nullptr, "-log_view", &flg));
  enabled = enabled || (flg == PETSC_TRUE);
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace

void Profiler::enable()
{
  enabled = true;
}

PetscErrorCode Profiler::register_stage(const char* name, PetscInt* id)
{
  PetscFunctionBeginUser;
  PetscCall(check_option());

  PetscLogStage stage;
  PetscCall(PetscLogStageGetId(name, &stage));
  if (stage < 0)
    PetscCall(PetscLogStageRegister(name, &stage));
  *id = stage;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Profiler::push_stage(PetscInt id)
{
  PetscFunctionBeginUser;
  if (!enabled)
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogStagePush(id));
  active_stages.push(id);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Profiler::pop_stage(PetscInt id)
{
  PetscFunctionBeginUser;
  if (!enabled)
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscAssert(!active_stages.empty() && active_stages.top() == id, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
    "Profiler::pop_stage(%d) does not match the last push", (int)id);

  PetscCall(PetscLogStagePop());
  active_stages.pop();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Profiler::register_event(const char* name, PetscInt* id)
{
  PetscFunctionBeginUser;
  PetscCall(check_option());

  PetscAssert(!omp_in_parallel(), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
    "Profiler::register_event(%s) must be called outside parallel regions", name);

  PetscLogEvent event;
  PetscCall(PetscLogEventGetId(name, &event));
  if (event < 0) {
    if (classid == 0)
      PetscCall(PetscClassIdRegister("Profiler", &classid));
    PetscCall(PetscLogEventRegister(name, classid, &event));
  }
  *id = event;

  // Slots are sized here, so that the threaded path never allocates shared data.
  slots.resize(omp_get_max_threads());
  for (ThreadSlot& slot : slots) {
    if (slot.events.size() <= (std::size_t)event)
      slot.events.resize(event + 1);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Profiler::push_event(PetscInt id)
{
  PetscFunctionBeginUser;
  if (!enabled)
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscAssert(!omp_in_parallel(), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
    "Profiler::push_event(%d) is serial only, use push_phase", (int)id);

  PetscCall(PetscLogEventBegin(id, 0, 0, 0, 0));
  active_events.push(id);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Profiler::pop_event(PetscInt id)
{
  PetscFunctionBeginUser;
  if (!enabled)
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscAssert(!omp_in_parallel(), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
    "Profiler::pop_event(%d) is serial only, use pop_phase", (int)id);

  PetscAssert(!active_events.empty() && active_events.top() == id, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
    "Profiler::pop_event(%d) does not match the last push", (int)id);

  PetscCall(PetscLogEventEnd(id, 0, 0, 0, 0));
  active_events.pop();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Profiler::push_phase(PetscInt id)
{
  PetscFunctionBeginHot;
  if (!enabled)
    PetscFunctionReturn(PETSC_SUCCESS);

  if (!omp_in_parallel())
    PetscFunctionReturn(push_event(id));

  ThreadSlot& slot = slots[omp_get_thread_num()];
  ThreadSlot::Acc& acc = slot.events[id];
  acc.team_size = omp_get_num_threads();
  slot.active.push(id);
  PetscCall(PetscTime(&acc.start));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Profiler::pop_phase(PetscInt id)
{
  PetscFunctionBeginHot;
  if (!enabled)
    PetscFunctionReturn(PETSC_SUCCESS);

  if (!omp_in_parallel())
    PetscFunctionReturn(pop_event(id));

  PetscLogDouble now;
  PetscCall(PetscTime(&now));

  ThreadSlot& slot = slots[omp_get_thread_num()];
  PetscAssert(!slot.active.empty() && slot.active.top() == id, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
    "Profiler::pop_phase(%d) does not match the last push in the thread", (int)id);

  ThreadSlot::Acc& acc = slot.events[id];
  acc.time += now - acc.start;
  acc.count++;
  slot.active.pop();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Profiler::flush()
{
  PetscFunctionBeginUser;
  PetscAssert(!omp_in_parallel(), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
    "Profiler::flush() must be called outside parallel regions");

  if (!enabled || slots.empty())
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscLogHandler handler;
  PetscCall(PetscLogGetDefaultHandler(&handler));

  PetscLogState state;
  PetscCall(PetscLogHandlerGetState(handler, &state));

  PetscLogStage stage;
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));

  for (std::size_t id = 0; id < slots[0].events.size(); id++) {
    PetscLogDouble time = 0;
    PetscInt count = 0;
    PetscInt team_size = 0;

    for (ThreadSlot& slot : slots) {
      ThreadSlot::Acc& acc = slot.events[id];
      PetscAssert(slot.active.empty(), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
        "Profiler::flush() called with an unmatched push_phase(%d)", (int)slot.active.top());

      time += acc.time;
      count += acc.count;
      team_size = std::max(team_size, acc.team_size);
      acc = {};
    }

    if (count == 0)
      continue;

    // Dividing by the team size, not by the number
    // of busy threads, makes idle threads lower the mean.
    PetscEventPerfInfo* info;
    PetscCall(PetscLogHandlerGetEventPerfInfo(handler, stage, (PetscLogEvent)id, &info));
    info->time += time / team_size;
    info->count += count;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
