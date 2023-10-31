#include "log.h"

#include <spdlog/sinks/basic_file_sink.h>

std::shared_ptr<spdlog::logger> Log::logger_;

void Log::Init(const std::string& filename) {
  logger_ = spdlog::basic_logger_mt("Plasma_simulation", filename, /* truncate = */ true);

  logger_->set_pattern("%^[%m/%d, %R, %L]%$ %v");

  logger_->set_level(spdlog::level::trace);
  logger_->flush_on(spdlog::level::warn);
}
