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
  PetscErrorCode push(std::string_view name);

  PetscErrorCode pop();

  PetscLogDouble get(PetscLogStage id);
  PetscLogDouble get(std::string_view name);

  PetscErrorCode log_timings(PetscInt skip = 0, PetscInt indent = 2) const;

private:
  using Item = std::pair<std::string_view, PetscLogDouble>;
  using Storage = std::vector<Item>;

  Storage::iterator times_find(std::string_view name);
  PetscLogDouble& times_at(std::string_view name);

  std::stack<std::string_view> active_;
  Storage times_;
};

#endif  // SRC_UTILS_SYNC_CLOCK_H
