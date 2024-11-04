#ifndef SRC_DIAGNOSTICS_FIELDS_ENERGY_H
#define SRC_DIAGNOSTICS_FIELDS_ENERGY_H

#include <petscdmda.h>
#include <petscvec.h>

#include "src/pch.h"
#include "src/interfaces/diagnostic.h"
#include "src/utils/sync_binary_file.h"

/// @todo Remove namespace here
namespace basic {

class Fields_energy : public interfaces::Diagnostic {
public:
  Fields_energy(const std::string& out_dir, DM da, Vec E, Vec B);

  PetscErrorCode diagnose(timestep_t t) override;

private:
  Sync_binary_file file_;

  DM da_;
  Vec E_;
  Vec B_;
};

}

#endif  // SRC_DIAGNOSTICS_FIELDS_ENERGY_H
