#ifndef SRC_UTILS_LOG_H
#define SRC_UTILS_LOG_H

#include <string>
#include <memory>

#include <mpi.h>
#include <spdlog/spdlog.h>

class Log {
public:
  static void init(const std::string& filename);

  template <typename... Args>
  static void trace(MPI_Comm comm, spdlog::format_string_t<Args...> fmt, Args&&... args);

  template <typename... Args>
  static void info(MPI_Comm comm, spdlog::format_string_t<Args...> fmt, Args&&... args);

  template <typename... Args>
  static void warn(MPI_Comm comm, spdlog::format_string_t<Args...> fmt, Args&&... args);

  template <typename... Args>
  static void error(MPI_Comm comm, spdlog::format_string_t<Args...> fmt, Args&&... args);

  static void flush(MPI_Comm comm);

private:
  template <typename... Args>
  static void log(MPI_Comm comm, spdlog::level::level_enum lvl, spdlog::format_string_t<Args...> fmt, Args&&... args);

  static std::shared_ptr<spdlog::logger> logger_;
};

#include "log.tpp"

#if LOGGING
#define LOG_INIT(filename) ::Log::init(filename)
#define LOG_TRACE(...)     ::Log::trace(MPI_COMM_WORLD, __VA_ARGS__)
#define LOG_INFO(...)      ::Log::info(MPI_COMM_WORLD, __VA_ARGS__)
#define LOG_WARN(...)      ::Log::warn(MPI_COMM_WORLD, __VA_ARGS__)
#define LOG_ERROR(...)     ::Log::error(MPI_COMM_WORLD, __VA_ARGS__)
#define LOG_FLUSH()        ::Log::flush(MPI_COMM_WORLD)

#else
#define LOG_INIT(filename)
#define LOG_TRACE(...)
#define LOG_INFO(...)
#define LOG_WARN(...)
#define LOG_ERROR(...)
#define LOG_FATAL(...)

#endif

#endif // SRC_UTILS_LOG_H
