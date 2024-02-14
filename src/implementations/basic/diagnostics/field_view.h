#ifndef SRC_BASIC_DIAGNOSTICS_FIELD_VIEW_H
#define SRC_BASIC_DIAGNOSTICS_FIELD_VIEW_H

#include "src/interfaces/diagnostic.h"

#include <petscdmda.h>
#include <petscvec.h>

#include "src/pch.h"
#include "src/utils/mpi_binary_file.h"

namespace basic {

class Field_view : public interfaces::Diagnostic {
public:
  Field_view(const std::string& result_directory, const DM da, const Vec field);
  PetscErrorCode diagnose(timestep_t t) override;

private:
  const DM da_;
  const Vec field_;

  MPI_binary_file file_;
};

}

#endif  // SRC_BASIC_DIAGNOSTICS_FIELD_VIEW_H
