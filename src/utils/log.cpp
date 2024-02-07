#include "log.h"

#include <spdlog/sinks/basic_file_sink.h>

std::shared_ptr<spdlog::logger> Log::logger_;

/* static */ void Log::init(MPI_Comm comm, const std::string& filename) {
  assert(comm != MPI_COMM_NULL && "Logger::inti() called with MPI_COMM_NULL");

  comm_ = comm;
  logger_ = spdlog::basic_logger_mt("basic_logger", filename);

  logger_->set_pattern("%^[%m/%d, %T, %L]%$ %v");

  logger_->set_level(spdlog::level::trace);
  logger_->flush_on(spdlog::level::warn);
}

/* static */ void Log::flush() {
  assert(MPI_COMM_WORLD != MPI_COMM_NULL && "Logger::flush() called with MPI_COMM_NULL");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) { logger_->flush(); }
}
