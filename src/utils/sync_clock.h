#ifndef SRC_UTILS_SYNC_CLOCK_H
#define SRC_UTILS_SYNC_CLOCK_H

#include <stack>

#include "src/pch.h"
#include "src/utils/sync_file.h"
#include "src/utils/utils.h"

class SyncClock {
public:
  DEFAULT_COPYABLE(SyncClock);

  SyncClock() = default;
  ~SyncClock() = default;

  PetscErrorCode push(PetscLogStage id);
  PetscErrorCode push(const std::string& name);

  PetscErrorCode pop();

  PetscLogDouble get(PetscLogStage id);
  PetscLogDouble get(const std::string& name);

private:
  std::stack<std::string> active_;
  std::map<std::string, PetscLogDouble> times_;
};

#endif  // SRC_UTILS_SYNC_CLOCK_H
