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

  virtual PetscErrorCode diagnose(PetscInt t) = 0;

  static std::string format_time(PetscInt t)
  {
    auto time_width = (PetscInt)std::to_string(geom_nt).size();
    std::stringstream ss;
    ss.width(time_width);
    ss.fill('0');
    ss << t;
    return ss.str();
  }

protected:
  std::string out_dir_;
};

}  // namespace interfaces

using Diagnostic_up = std::unique_ptr<interfaces::Diagnostic>;

#endif  // SRC_INTERFACES_DIAGNOSTIC_H
