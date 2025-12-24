#ifndef SRC_DIAGNOSTICS_ENERGY_H
#define SRC_DIAGNOSTICS_ENERGY_H

#include "src/pch.h"
#include "src/interfaces/simulation.h"
#include "src/diagnostics/utils/table_diagnostic.h"

class Energy : public interfaces::Diagnostic {
public:
  Energy(const interfaces::Simulation& simulation);

  PetscErrorCode diagnose(PetscInt t) override;

  static PetscReal get_field(const Vector3R& f);
  static PetscReal get_kinetic(const Vector3R& p, PetscReal m, PetscInt Np);

protected:
  PetscErrorCode calculate_field();
  PetscErrorCode calculate_kinetic();
  PetscErrorCode calculate_spectral();

  PetscErrorCode fill_energy(PetscInt t);
  virtual PetscErrorCode fill_energy_cons(PetscInt t);

  const interfaces::Simulation& simulation;
  TableDiagnostic energy;
  TableDiagnostic energy_cons;

  PetscReal E = 0, E0 = 0, dE = 0, std_E = 0;
  PetscReal B = 0, B0 = 0, dB = 0, std_B = 0;
  PetscReal dF = 0, dK = 0;

  std::vector<PetscReal> K, K0, std_K;

  const PetscInt kmax = 2;
  std::vector<PetscReal> Ek, Bk;
};

#endif  // SRC_DIAGNOSTICS_ENERGY_H
