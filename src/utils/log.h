#ifndef SRC_UTILS_LOG_H
#define SRC_UTILS_LOG_H

#include <string>
#include <memory>

#include <spdlog/spdlog.h>

class Log {
 public:
  static void init(const std::string& filename);
  inline static std::shared_ptr<spdlog::logger>& get_logger() { return logger_; }

 private:
  static std::shared_ptr<spdlog::logger> logger_;
};


#if LOGGING
#define LOG_INIT(filename) ::Log::init(filename)
#define LOG_TRACE(...)     ::Log::get_logger()->trace(__VA_ARGS__)
#define LOG_INFO(...)      ::Log::get_logger()->info(__VA_ARGS__)
#define LOG_WARN(...)      ::Log::get_logger()->warn(__VA_ARGS__)
#define LOG_ERROR(...)     ::Log::get_logger()->error(__VA_ARGS__)
#define LOG_FATAL(...)     ::Log::get_logger()->critical(__VA_ARGS__)
#define LOG_FLUSH()        ::Log::get_logger()->flush()

#else
#define LOG_INIT(filename)
#define LOG_TRACE(...)
#define LOG_INFO(...)
#define LOG_WARN(...)
#define LOG_ERROR(...)
#define LOG_FATAL(...)

#endif

#endif // SRC_UTILS_LOG_H
