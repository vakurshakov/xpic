#ifndef SRC_BASIC_DIAGNOSTICS_FIELDS_ENERGY_H
#define SRC_BASIC_DIAGNOSTICS_FIELDS_ENERGY_H

#include "src/interfaces/diagnostic.h"

#include <petscdmda.h>
#include <petscvec.h>

#include "src/pch.h"
#include "src/utils/sync_binary_file.h"

namespace basic {

class Fields_energy : public interfaces::Diagnostic {
public:
  Fields_energy(const std::string& result_directory, DM da, Vec E, Vec B);

  PetscErrorCode diagnose(timestep_t t) override;

private:
  Sync_binary_file file_;

  DM da_;
  Vec E_;
  Vec B_;
};

}

#endif  // SRC_BASIC_DIAGNOSTICS_FIELDS_ENERGY_H
