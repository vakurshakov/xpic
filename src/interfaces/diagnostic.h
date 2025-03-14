#ifndef SRC_INTERFACES_DIAGNOSTIC_H
#define SRC_INTERFACES_DIAGNOSTIC_H

#include "src/pch.h"
#include "src/utils/utils.h"

namespace interfaces {

class Diagnostic {
public:
  DEFAULT_MOVABLE(Diagnostic);

  Diagnostic();
  Diagnostic(const std::string& out_dir);

  virtual ~Diagnostic() = default;

  /// @brief The main method to override
  virtual PetscErrorCode diagnose(PetscInt t) = 0;

  /// @returns The string padded with the zeros of `geom_nt` width
  static std::string format_time(PetscInt t);

protected:
  std::string out_dir_;
  PetscInt diagnose_period_ ;
};

}  // namespace interfaces

using Diagnostic_up = std::unique_ptr<interfaces::Diagnostic>;

#endif  // SRC_INTERFACES_DIAGNOSTIC_H
