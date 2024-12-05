#ifndef SRC_INTERFACES_DIAGNOSTIC_H
#define SRC_INTERFACES_DIAGNOSTIC_H

#include "src/pch.h"

namespace interfaces {

class Diagnostic {
public:
  virtual ~Diagnostic() = default;

  Diagnostic(const std::string& out_dir)
    : out_dir_(out_dir)
  {
  }

  virtual PetscErrorCode diagnose(timestep_t t) = 0;

protected:
  std::string out_dir_;
};

}  // namespace interfaces

using Diagnostic_up = std::unique_ptr<interfaces::Diagnostic>;

#endif  // SRC_INTERFACES_DIAGNOSTIC_H
