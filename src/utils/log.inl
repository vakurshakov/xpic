#include "log.h"

#include <cassert>

template <typename... Args>
/* static */ void Log::log(spdlog::level::level_enum lvl, spdlog::format_string_t<Args...> fmt, Args &&...args) {
  int rank;
  MPI_Comm_rank(comm_, &rank);
  if (rank == 0) { logger_->log(lvl, fmt, std::forward<Args>(args)...); }
}
