#ifndef SRC_UTILS_LOG_H
#define SRC_UTILS_LOG_H

#include <string>
#include <memory>

#include <mpi.h>
#include <spdlog/spdlog.h>

class Log {
public:
  static void init(MPI_Comm comm, const std::string& filename);

  template <typename... Args>
  static void log(spdlog::level::level_enum lvl, spdlog::format_string_t<Args...> fmt, Args&&... args);

  static void flush();

private:
  static MPI_Comm comm_;
  static std::shared_ptr<spdlog::logger> logger_;
};

#include "log.inl"

#if LOGGING
#define LOG_INIT(comm, filename) ::Log::init(comm, filename)
#define LOG_TRACE(...)           ::Log::log(spdlog::level::trace, __VA_ARGS__)
#define LOG_INFO(...)            ::Log::log(spdlog::level::info,  __VA_ARGS__)
#define LOG_WARN(...)            ::Log::log(spdlog::level::warn,  __VA_ARGS__)
#define LOG_ERROR(...)           ::Log::log(spdlog::level::error, __VA_ARGS__)
#define LOG_FATAL(...)           ::Log::log(spdlog::level::fatal, __VA_ARGS__)
#define LOG_FLUSH()              ::Log::flush()

#else
#define LOG_INIT(filename)
#define LOG_TRACE(...)
#define LOG_INFO(...)
#define LOG_WARN(...)
#define LOG_ERROR(...)
#define LOG_FATAL(...)

#endif

#endif // SRC_UTILS_LOG_H
