#ifndef SRC_UTILS_TIME_MANAGER_H
#define SRC_UTILS_TIME_MANAGER_H

// Basic instrumentation profiler by Cherno.

#include <omp.h>

#include <string>
#include <chrono>
#include <algorithm>
#include <fstream>

struct Profile_result {
  std::string name;   // recorded profile;
  std::string start;  // profile beginning in POSIX format;
  int64_t duration;   // elapsed time in microseconds;
  int thread_num;     // thread number given by omp call.
};


class Instrumentor {
 public:
  Instrumentor();

  void begin_session(const std::string& filepath = "results.json");
  void end_session();

  void write_profile(Profile_result&& result);

  static Instrumentor& get();

 private:
  std::ofstream output_stream_;
  int profile_count_;

  const bool pretty_write_ = false;
  void write_header();
  void write_footer();
};


class Instrumentation_timer {
 public:
  Instrumentation_timer(const char* name);
  virtual ~Instrumentation_timer();

  virtual void stop();

 protected:
  void write_profile(const std::chrono::microseconds& duration) const;

  bool stopped_;
  const char* name_;
  std::chrono::time_point<std::chrono::system_clock> start_;
};

class Accumulative_timer : public Instrumentation_timer {
 public:
  Accumulative_timer(const char* name);
  ~Accumulative_timer() override = default;

  void start(int number_of_iterations);
  void stop() override;

 private:
  void set_new(int number_of_iterations);
  void drop_and_reset();

  const char* name_;

  int current_ = 0;
  int number_of_iterations_;
  std::chrono::microseconds accumulated_duration;
  std::chrono::time_point<std::chrono::system_clock> local_start_;
};


#if TIME_PROFILING

// One level of macro indirection is required in order
// to resolve __LINE__ and get varN instead of var__LINE__.
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a ## b

#define BEGIN_SESSION(filepath)  ::Instrumentor::get().begin_session(filepath)
#define END_SESSION()            ::Instrumentor::get().end_session()
#define PROFILE_SCOPE(name)      ::Instrumentation_timer CONCAT(timer, __LINE__)(name)
#define PROFILE_FUNCTION()       PROFILE_SCOPE(__PRETTY_FUNCTION__)

#define ACCUMULATIVE_PROFILE(name, n, call_to_profile)              \
  thread_local ::Accumulative_timer CONCAT(timer, __LINE__)(name);  \
  CONCAT(timer, __LINE__).start(n);                                 \
  call_to_profile;                                                  \
  CONCAT(timer, __LINE__).stop()                                    \

#else
#define BEGIN_SESSION(filepath)
#define END_SESSION()
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()

#define ACCUMULATIVE_PROFILE(name, n, call_to_profile)  \
  call_to_profile;                                      \

#endif

#endif  // SRC_UTILS_TIME_MANAGER_H
