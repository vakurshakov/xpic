#include "log.h"

#include <cassert>

template <typename... Args>
/* static */ void Log::log(MPI_Comm comm, spdlog::level::level_enum lvl, spdlog::format_string_t<Args...> fmt, Args &&...args) {
  assert(comm != MPI_COMM_NULL && "Logger::log() called with MPI_COMM_NULL");

  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0) { logger_->log(lvl, fmt, std::forward<Args>(args)...); }
}
