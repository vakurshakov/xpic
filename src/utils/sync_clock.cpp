#include "sync_clock.h"

PetscErrorCode SyncClock::push(PetscLogStage id)
{
  PetscFunctionBeginUser;
  const char* name;
  PetscCall(PetscLogStageGetName(id, &name));
  PetscCall(push(name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SyncClock::push(const std::string& name)
{
  PetscFunctionBeginUser;
  auto it = times_.find(name);
  if (it == times_.end())
    it = times_.insert(it, std::pair(name, 0.0));

  active_.push(name);
  PetscCall(PetscTime(&it->second));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SyncClock::pop()
{
  PetscFunctionBeginUser;
  const auto& name = active_.top();

  PetscLogDouble time;
  PetscCall(PetscTime(&time));

  PetscLogDouble& st = times_.at(name);
  st = time - st;

  PetscCall(MPI_Allreduce(MPI_IN_PLACE, &st, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_WORLD));

  PetscMPIInt size;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  st /= size;

  active_.pop();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscLogDouble SyncClock::get(PetscLogStage id)
{
  PetscFunctionBeginUser;
  const char* name;
  PetscCall(PetscLogStageGetName(id, &name));
  PetscFunctionReturn(get(name));
}

PetscLogDouble SyncClock::get(const std::string& name)
{
  PetscFunctionBeginUser;
  auto it = times_.find(name);
  PetscFunctionReturn(it != times_.end() ? it->second : 0.0);
}
