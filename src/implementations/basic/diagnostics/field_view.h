#ifndef SRC_BASIC_DIAGNOSTICS_FIELD_VIEW_H
#define SRC_BASIC_DIAGNOSTICS_FIELD_VIEW_H

#include "src/interfaces/diagnostic.h"

#include <petscdmda.h>
#include <petscvec.h>

#include "src/pch.h"
#include "src/utils/mpi_binary_file.h"
#include "src/vectors/vector_classes.h"

namespace basic {

class Field_view : public interfaces::Diagnostic {
public:
  struct Region {
    // Provided by DMDAGetInfo(da_, &ndim, ...)
    static constexpr PetscInt ndim = 4;
    PetscInt start[ndim];
    PetscInt size[ndim];
  };

  Field_view(MPI_Comm comm, const std::string& result_directory, const DM& da, const Vec& field);

  PetscErrorCode set_diagnosed_region(const Region& region);
  PetscErrorCode diagnose(timestep_t t) override;

private:
  const DM& da_;
  const Vec& field_;

  MPI_Comm comm_;
  MPI_binary_file file_;
};

}

#endif  // SRC_BASIC_DIAGNOSTICS_FIELD_VIEW_H
