#ifndef SRC_BASIC_DIAGNOSTICS_FIELD_VIEW_H
#define SRC_BASIC_DIAGNOSTICS_FIELD_VIEW_H

#include "src/interfaces/diagnostic.h"

#include <petscdmda.h>
#include <petscvec.h>

#include "src/pch.h"
#include "src/utils/mpi_binary_file.h"

#include "src/utils/vector4.h"


class Field_view : public interfaces::Diagnostic {
public:
  struct Region {
    PetscInt dim;
    PetscInt dof;
    Vector4I start;
    Vector4I size;
  };

  /**
   * @brief Constructs `Field_view` diagnostic of a particular `field`.
   * @note Result _can_ be `nullptr`, if `region` doesn't touch the local part of DM.
   */
  static std::unique_ptr<Field_view> create(const std::string& out_dir, DM da, Vec field, const Region& region);

  PetscErrorCode diagnose(timestep_t t) override;

protected:
  static PetscErrorCode get_local_communicator(DM da, const Region& region, MPI_Comm* newcomm);

  Field_view(const std::string& out_dir, DM da, Vec field, MPI_Comm newcomm);
  PetscErrorCode set_data_views(const Region& region);

  DM da_;
  Vec field_;
  Region region_;

  MPI_Comm comm_;
  MPI_binary_file file_;
};

#endif  // SRC_BASIC_DIAGNOSTICS_FIELD_VIEW_H
