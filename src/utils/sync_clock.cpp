#include "sync_clock.h"

PetscErrorCode SyncClock::push(PetscLogStage id)
{
  PetscFunctionBeginUser;
  const char* name;
  PetscCall(PetscLogStageGetName(id, &name));
  PetscCall(push(name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SyncClock::push(std::string_view name)
{
  PetscFunctionBeginUser;
  auto it = times_find(name);
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

  PetscLogDouble& st = times_at(name);
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

PetscLogDouble SyncClock::get(std::string_view name)
{
  PetscFunctionBeginUser;
  auto it = times_find(name);
  PetscFunctionReturn(it != times_.end() ? it->second : 0.0);
}

SyncClock::Storage::iterator SyncClock::times_find(std::string_view name)
{
  return std::find_if(times_.begin(), times_.end(), [name](const Item& item) {
    return item.first == name;
  });
}

PetscLogDouble& SyncClock::times_at(std::string_view name)
{
  auto it = times_find(name);
  PetscCheckAbort(it != times_.end(), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Name wasn't specified before accessing it, %s", name.data());
  return it->second;
}

PetscErrorCode SyncClock::log_timings(PetscInt skip, PetscInt indent) const
{
  PetscFunctionBeginUser;
  PetscInt size = (PetscInt)times_.size();
  PetscLogDouble sum = 0.0;

  for (PetscInt i = skip; i < size; ++i) {
    sum += times_[i].second;
  }

  LOG("{:{}s}Summary of Stages:  ------- Time -------", " ", indent);
  LOG("{:{}s}                        Avg         %", " ", indent);

  for (PetscInt i = skip; i < size; ++i) {
    auto& [name, time] = times_[i];
    LOG("{:{}s}{:2d}: {:>15s}: {:6.4e}  {:5.1f}%", " ", indent, i, name, time, 100.0 * time / sum);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
