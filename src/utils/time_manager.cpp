#include "time_manager.h"

namespace ch = std::chrono;

Instrumentor::Instrumentor()
  : profile_count_(0) {}

void Instrumentor::begin_session(const std::string& filepath) {
  output_stream_.open(filepath);
  write_header();
}

void Instrumentor::end_session() {
  write_footer();

  output_stream_.close();
  profile_count_ = 0;
}

void Instrumentor::write_profile(Profile_result&& result) {
  std::string name = result.name;
  std::replace(name.begin(), name.end(), '"', '\'');

  if (profile_count_++ > 0)
    output_stream_ << ",\n";

  if (pretty_write_) {
    output_stream_ << "  {\n";
    output_stream_ << "    \"name\": \"" << name << "\",\n";
    output_stream_ << "    \"start\": " << result.start << ",\n";
    output_stream_ << "    \"dur\": " << result.duration << ",\n";
    output_stream_ << "    \"tid\": " << result.thread_num << "\n  }";
  }
  else {
    output_stream_ << "{\"name\":\"" << name << "\",";
    output_stream_ << "\"start\":" << result.start << ",";
    output_stream_ << "\"dur\":" << result.duration << ",";
    output_stream_ << "\"tid\":" << result.thread_num << "}";
  }
  output_stream_.flush();
}

/* static */ Instrumentor& Instrumentor::get() {
  static Instrumentor instance;
  return instance;
}

void Instrumentor::write_header() {
  output_stream_ << (pretty_write_ ? "{ \"record\": [\n" : "{\"record\":[\n");
  output_stream_.flush();
}

void Instrumentor::write_footer() {
  output_stream_ << "\n]}";
  output_stream_.flush();
}


Instrumentation_timer::Instrumentation_timer(const char* name)
  : stopped_(false), name_(name) {
  start_ = ch::system_clock::now();
}

Instrumentation_timer::~Instrumentation_timer() {
  if (!stopped_)
   stop();
}

void Instrumentation_timer::stop() {
  const auto end = ch::system_clock::now();
  const auto duration = ch::duration_cast<ch::microseconds>(end - start_);

  #pragma omp critical
  write_profile(duration);

  stopped_ = true;
}

void Instrumentation_timer::write_profile(const ch::microseconds& duration) const {
  int thread_num = omp_get_thread_num();

  Instrumentor::get().write_profile({
    name_,
    std::to_string(start_.time_since_epoch().count() / 1e9),  // from ns
    duration.count(),
    thread_num
  });
}


Accumulative_timer::Accumulative_timer(const char* name)
  : Instrumentation_timer(name) {}

void Accumulative_timer::start(int number_of_iterations) {
  if (stopped_) {
    set_new(number_of_iterations);
    stopped_ = false;
  }

  current_++;
  local_start_ = std::chrono::system_clock::now();
}

void Accumulative_timer::set_new(int number_of_iterations) {
  // If number_of_iterations is less then num_threads,
  // timer will drop_and_reset after the first iteration.
  number_of_iterations_ = number_of_iterations / omp_get_num_threads();

  start_ = ch::system_clock::now();
  accumulated_duration = accumulated_duration.zero();
}

void Accumulative_timer::stop() {
  const auto now = ch::system_clock::now();
  accumulated_duration += ch::duration_cast<ch::microseconds>(now - local_start_);

  if (current_ >= number_of_iterations_)
    drop_and_reset();
}

void Accumulative_timer::drop_and_reset() {
  #pragma omp critical
  write_profile(accumulated_duration);

  stopped_ = true;
  current_ = 0;
}
