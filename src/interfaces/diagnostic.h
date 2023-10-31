#ifndef SRC_INTERFACES_DIAGNOSTIC_H
#define SRC_INTERFACES_DIAGNOSTIC_H

#include "src/pch.h"
#include "src/utils/binary_file.h"

class Diagnostic {
 public:
  virtual ~Diagnostic() = default;

  Diagnostic(const std::string_view& result_directory)
    : result_directory_(result_directory) {}

  virtual void diagnose(timestep_t t) = 0;

 protected:
  std::string result_directory_;
  Binary_file result_file_;
};

#endif  // SRC_INTERFACES_DIAGNOSTIC_H
