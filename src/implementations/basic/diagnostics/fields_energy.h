#ifndef SRC_BASIC_DIAGNOSTICS_FIELDS_ENERGY_H
#define SRC_BASIC_DIAGNOSTICS_FIELDS_ENERGY_H

#include "src/interfaces/diagnostic.h"

#include <petscdmda.h>
#include <petscvec.h>

#include "src/pch.h"
#include "src/utils/binary_file.h"

namespace basic {

class Fields_energy : public interfaces::Diagnostic {
public:
  Fields_energy(const std::string& result_directory, const DM da, const Vec E, const Vec B);
  PetscErrorCode diagnose(timestep_t t) override;

private:
  Binary_file file_;

  const DM da_;
  const Vec E_;
  const Vec B_;
};

}

#endif  // SRC_BASIC_DIAGNOSTICS_FIELDS_ENERGY_H
