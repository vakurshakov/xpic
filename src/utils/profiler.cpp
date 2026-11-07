#include "profiler.h"

#include <petsclog.h>

namespace {

struct Entry {
  std::string name;
  bool is_stage;
  PetscInt petsc_id;  // `PetscLogStage` or `PetscLogEvent`
};

std::vector<Entry> entries;
std::vector<PetscInt> active;  // handles of the pushed entries
PetscClassId classid = 0;
bool enabled = false;
bool option_checked = false;

PetscErrorCode register_entry(const char* name, bool is_stage, PetscInt* id)
{
  PetscFunctionBeginUser;
  // PETSc is the source of truth for names: look the name up, register if missing
  PetscInt petsc_id;
  if (is_stage) {
    PetscLogStage stage;
    PetscCall(PetscLogStageGetId(name, &stage));
    if (stage < 0)
      PetscCall(PetscLogStageRegister(name, &stage));
    petsc_id = stage;
  }
  else {
    PetscLogEvent event;
    PetscCall(PetscLogEventGetId(name, &event));
    if (event < 0) {
      if (classid == 0)
        PetscCall(PetscClassIdRegister("Profiler", &classid));
      PetscCall(PetscLogEventRegister(name, classid, &event));
    }
    petsc_id = event;
  }

  for (std::size_t i = 0; i < entries.size(); i++) {
    if (entries[i].is_stage == is_stage && entries[i].petsc_id == petsc_id) {
      *id = (PetscInt)i;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }

  entries.push_back({name, is_stage, petsc_id});
  *id = (PetscInt)entries.size() - 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace

PetscErrorCode Profiler::register_stage(const char* name, PetscInt* id)
{
  return register_entry(name, true, id);
}

PetscErrorCode Profiler::register_event(const char* name, PetscInt* id)
{
  return register_entry(name, false, id);
}

void Profiler::enable()
{
  enabled = true;
  option_checked = true;
}

PetscErrorCode Profiler::push(PetscInt id)
{
  PetscFunctionBeginUser;
  if (!option_checked) {
    PetscBool flg;
    PetscCall(PetscOptionsHasName(nullptr, nullptr, "-log_view", &flg));
    enabled = (flg == PETSC_TRUE);
    option_checked = true;
  }
  if (!enabled)
    PetscFunctionReturn(PETSC_SUCCESS);

  const Entry& entry = entries.at(id);
  if (entry.is_stage)
    PetscCall(PetscLogStagePush(entry.petsc_id));
  else
    PetscCall(PetscLogEventBegin(entry.petsc_id, 0, 0, 0, 0));

  active.push_back(id);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Profiler::pop(PetscInt id)
{
  PetscFunctionBeginUser;
  if (!enabled)
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscCheckAbort(!active.empty() && active.back() == id, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
    "Profiler::pop(%s) does not match the last push (%s)", entries.at(id).name.c_str(),
    active.empty() ? "none" : entries[active.back()].name.c_str());

  const Entry& entry = entries[id];
  if (entry.is_stage)
    PetscCall(PetscLogStagePop());
  else
    PetscCall(PetscLogEventEnd(entry.petsc_id, 0, 0, 0, 0));

  active.pop_back();
  PetscFunctionReturn(PETSC_SUCCESS);
}
