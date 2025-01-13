#ifndef SRC_DIAGNOSTICS_FIELDS_ENERGY_H
#define SRC_DIAGNOSTICS_FIELDS_ENERGY_H

#include <petscdmda.h>
#include <petscvec.h>

#include "src/pch.h"
#include "src/interfaces/diagnostic.h"
#include "src/utils/sync_binary_file.h"
#include "src/utils/vector3.h"

class FieldsEnergy : public interfaces::Diagnostic {
public:
  FieldsEnergy(DM da, Vec E, Vec B);
  FieldsEnergy(const std::string& out_dir, DM da, Vec E, Vec B);

  PetscErrorCode diagnose(timestep_t t) override;

  PetscErrorCode calculate_energies();
  PetscReal get_electric_energy() const;
  PetscReal get_magnetic_energy() const;

private:
  SyncBinaryFile file_;

  DM da_;
  Vec E_;
  Vec B_;

  Vector3R energy_E_;
  Vector3R energy_B_;
};

#endif  // SRC_DIAGNOSTICS_FIELDS_ENERGY_H
