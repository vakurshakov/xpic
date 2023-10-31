#ifndef SRC_UTILS_LOG_H
#define SRC_UTILS_LOG_H

#include <string>
#include <memory>

#include <spdlog/spdlog.h>

class Log {
 public:
  static void Init(const std::string& filename);

  inline static std::shared_ptr<spdlog::logger>& GetLogger() { return logger_; }

 private:
  static std::shared_ptr<spdlog::logger> logger_;
};


#if LOGGING
#define LOG_INIT(filename) ::Log::Init(filename)
#define LOG_TRACE(...)     ::Log::GetLogger()->trace(__VA_ARGS__)
#define LOG_INFO(...)      ::Log::GetLogger()->info(__VA_ARGS__)
#define LOG_WARN(...)      ::Log::GetLogger()->warn(__VA_ARGS__)
#define LOG_ERROR(...)     ::Log::GetLogger()->error(__VA_ARGS__)
#define LOG_FATAL(...)     ::Log::GetLogger()->critical(__VA_ARGS__)
#define LOG_FLUSH()        ::Log::GetLogger()->flush()

#else
#define LOG_INIT(filename)
#define LOG_TRACE(...)
#define LOG_INFO(...)
#define LOG_WARN(...)
#define LOG_ERROR(...)
#define LOG_FATAL(...)

#endif

#endif // SRC_UTILS_LOG_H
