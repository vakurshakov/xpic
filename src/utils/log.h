#ifndef SRC_UTILS_LOG_H
#define SRC_UTILS_LOG_H

#include <string>
#include <memory>

#include <mpi.h>
#include <spdlog/spdlog.h>

class Log {
private:
  template<typename... Args>
  using format_t = spdlog::format_string_t<Args...>;

public:
  static void init(const std::string& filename);

  template<typename... Args>
  static void trace(MPI_Comm comm, format_t<Args...> fmt, Args&&... args) {
    log(comm, spdlog::level::trace, fmt, std::forward<Args>(args)...);
  }

  template<typename... Args>
  static void info(MPI_Comm comm, format_t<Args...> fmt, Args&&... args) {
    log(comm, spdlog::level::info, fmt, std::forward<Args>(args)...);
  }

  template<typename... Args>
  static void warn(MPI_Comm comm, format_t<Args...> fmt, Args&&... args) {
    log(comm, spdlog::level::warn, fmt, std::forward<Args>(args)...);
  }

  template<typename... Args>
  static void error(MPI_Comm comm, format_t<Args...> fmt, Args&&... args) {
    log(comm, spdlog::level::err, fmt, std::forward<Args>(args)...);
  }

  static void flush(MPI_Comm comm);

private:
  template<typename... Args>
  static void log(MPI_Comm comm, spdlog::level::level_enum lvl, format_t<Args...> fmt, Args &&...args) {
    assert(comm != MPI_COMM_NULL && "Logger::log() called with MPI_COMM_NULL");

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) { logger_->log(lvl, fmt, std::forward<Args>(args)...); }
  }

  static std::shared_ptr<spdlog::logger> logger_;
};

#if LOGGING
#define LOG_INIT(filename) ::Log::init(filename)
#define LOG_TRACE(...)     ::Log::trace(PETSC_COMM_WORLD, __VA_ARGS__)
#define LOG_INFO(...)      ::Log::info(PETSC_COMM_WORLD, __VA_ARGS__)
#define LOG_WARN(...)      ::Log::warn(PETSC_COMM_WORLD, __VA_ARGS__)
#define LOG_ERROR(...)     ::Log::error(PETSC_COMM_WORLD, __VA_ARGS__)
#define LOG_FLUSH()        ::Log::flush(PETSC_COMM_WORLD)

#else
#define LOG_INIT(filename)
#define LOG_TRACE(...)
#define LOG_INFO(...)
#define LOG_WARN(...)
#define LOG_ERROR(...)
#define LOG_FATAL(...)

#endif

#endif // SRC_UTILS_LOG_H
