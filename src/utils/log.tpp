#include "log.h"

#include <cassert>

template <typename... Args>
/* static */ void Log::trace(MPI_Comm comm, spdlog::format_string_t<Args...> fmt, Args&&... args) {
  log(comm, spdlog::level::trace, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
/* static */ void Log::info(MPI_Comm comm, spdlog::format_string_t<Args...> fmt, Args&&... args) {
  log(comm, spdlog::level::info, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
/* static */ void Log::warn(MPI_Comm comm, spdlog::format_string_t<Args...> fmt, Args&&... args) {
  log(comm, spdlog::level::warn, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
/* static */ void Log::error(MPI_Comm comm, spdlog::format_string_t<Args...> fmt, Args&&... args) {
  log(comm, spdlog::level::err, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
/* static */ void Log::log(MPI_Comm comm, spdlog::level::level_enum lvl, spdlog::format_string_t<Args...> fmt, Args &&...args) {
  assert(comm != MPI_COMM_NULL && "Logger::log() called with MPI_COMM_NULL");

  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0) { logger_->log(lvl, fmt, std::forward<Args>(args)...); }
}
