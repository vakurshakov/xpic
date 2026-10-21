#ifndef SRC_DIAGNOSTICS_FIELD_VIEW_H
#define SRC_DIAGNOSTICS_FIELD_VIEW_H

#include <petscdmda.h>
#include <petscvec.h>

#include "src/pch.h"
#include "src/interfaces/diagnostic.h"
#include "src/utils/world.h"

class FieldView : public interfaces::Diagnostic {
public:
  /**
   * @brief Constructs `Field_view` diagnostic of a particular `field`.
   * @note Result _can_ be `nullptr`, if `region` doesn't touch
   * the local part of DM.
   * @todo Replace this with virtual `initialize()` method
   */
  static std::unique_ptr<FieldView> create(
    const std::string& out_dir, DM da, Vec field, const Region& region);

  PetscErrorCode finalize() override;
  PetscErrorCode diagnose(PetscInt t) override;

protected:
  FieldView(DM da, Vec field);
  FieldView(const std::string& out_dir, DM da, Vec field, MPI_Comm newcomm);

  virtual PetscErrorCode set_data_views(const Region& region);

  PetscErrorCode create_subarray(PetscInt ndim, const PetscInt sizes[],
    const PetscInt subsizes[], const PetscInt starts[], MPI_Datatype* type);

  PetscErrorCode open(const std::string& filename);
  PetscErrorCode flush();
  PetscErrorCode close();
  PetscErrorCode write(PetscInt size, const PetscReal* data);
  PetscErrorCode write_floats(const float* data);

  DM da;
  Vec field;
  Region region;

  MPI_Comm comm = MPI_COMM_NULL;
  MPI_File file = MPI_FILE_NULL;

  MPI_Datatype memview = MPI_DATATYPE_NULL;
  MPI_Datatype fileview = MPI_DATATYPE_NULL;
};

#endif  // SRC_DIAGNOSTICS_FIELD_VIEW_H
