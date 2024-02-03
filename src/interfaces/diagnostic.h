#ifndef SRC_INTERFACES_DIAGNOSTIC_H
#define SRC_INTERFACES_DIAGNOSTIC_H

#include "src/pch.h"

namespace interfaces {

class Diagnostic {
public:
  virtual ~Diagnostic() = default;

  Diagnostic(const std::string& result_directory)
    : result_directory_(result_directory) {}

  virtual PetscErrorCode diagnose(timestep_t t) = 0;

protected:
  std::string result_directory_;
};

}

#endif  // SRC_INTERFACES_DIAGNOSTIC_H
