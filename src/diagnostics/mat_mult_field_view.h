#ifndef SRC_DIAGNOSTICS_MAT_MULT_FIELD_VIEW_H
#define SRC_DIAGNOSTICS_MAT_MULT_FIELD_VIEW_H

#include "src/diagnostics/field_view.h"

/// @brief Writes `op * source` computed at each diagnosed step.
class MatMultFieldView : public FieldView {
public:
  /// @param result_da DM of the `op * source` result, may differ from the
  /// source one (e.g. for divergence).
  static std::unique_ptr<MatMultFieldView> create(const std::string& out_dir,
    DM result_da, Mat op, Vec source, const Region& region);

  PetscErrorCode finalize() override;
  PetscErrorCode diagnose(PetscInt t) override;

protected:
  MatMultFieldView(const std::string& out_dir, DM result_da, Vec result, Mat op,
    Vec source, MPI_Comm newcomm);

  Mat op = nullptr;
  Vec source = nullptr;
};

#endif  // SRC_DIAGNOSTICS_MAT_MULT_FIELD_VIEW_H
