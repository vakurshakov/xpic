#ifndef SRC_DIAGNOSTICS_ENERGY_H
#define SRC_DIAGNOSTICS_ENERGY_H

#include <petscvec.h>

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/diagnostics/utils/table_diagnostic.h"

class Energy : public TableDiagnostic {
  friend class EnergyConservation;

public:
  Energy(Vec E, Vec B, std::vector<const interfaces::Particles*> particles);

  PetscErrorCode initialize() override;
  PetscErrorCode add_columns(PetscInt t) override;
  PetscErrorCode calculate_energies();

  PetscReal get_electric_energy() const;
  PetscReal get_magnetic_energy() const;
  std::vector<PetscReal> get_kinetic_energies() const;

  static PetscReal get_field(const Vector3R& f);
  static PetscReal get_kinetic(const Vector3R& p, PetscReal m, PetscInt Np);

private:
  Vec E;
  Vec B;
  std::vector<const interfaces::Particles*> particles;

  PetscReal w_E = 0.0, std_E = 0.0;
  PetscReal w_B = 0.0, std_B = 0.0;
  std::vector<PetscReal> w_K, std_K;
};

#endif  // SRC_DIAGNOSTICS_ENERGY_H
