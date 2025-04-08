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
  Diagnostic(const std::string& out_dir, PetscInt diagnose_period);

  /// @brief Explicit finalize should be called to end up the diagnostic.
  virtual PetscErrorCode finalize()
  {
    PetscFunctionBeginUser;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /// @brief The main method to override
  virtual PetscErrorCode diagnose(PetscInt t) = 0;

protected:
  /// @returns The string padded with the zeros of `geom_nt` width
  static std::string format_time(PetscInt t);

  std::string out_dir_;
  PetscInt diagnose_period_;
};

}  // namespace interfaces

using Diagnostic_up = std::unique_ptr<interfaces::Diagnostic>;

#endif  // SRC_INTERFACES_DIAGNOSTIC_H
