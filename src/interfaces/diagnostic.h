#ifndef SRC_INTERFACES_DIAGNOSTIC_H
#define SRC_INTERFACES_DIAGNOSTIC_H

#include "src/pch.h"
#include "src/utils/utils.h"

namespace interfaces {

class Diagnostic {
public:
  DEFAULT_MOVABLE(Diagnostic);

  Diagnostic() = default;

  Diagnostic(const std::string& out_dir)
    : out_dir_(out_dir)
  {
  }

  virtual ~Diagnostic() = default;

  virtual PetscErrorCode diagnose(timestep_t t) = 0;

protected:
  std::string format_time(timestep_t t)
  {
    auto time_width = (PetscInt)std::to_string(geom_nt).size();
    std::stringstream ss;
    ss.width(time_width);
    ss.fill('0');
    ss << t;
    return ss.str();
  }

  std::string out_dir_;
};

}  // namespace interfaces

using Diagnostic_up = std::unique_ptr<interfaces::Diagnostic>;

#endif  // SRC_INTERFACES_DIAGNOSTIC_H
