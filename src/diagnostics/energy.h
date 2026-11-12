#ifndef SRC_DIAGNOSTICS_ENERGY_H
#define SRC_DIAGNOSTICS_ENERGY_H

#include "src/pch.h"
#include "src/interfaces/simulation.h"
#include "src/diagnostics/table_diagnostic.h"

class Energy : public interfaces::Diagnostic {
public:
  Energy(const interfaces::Simulation& simulation);

  PetscErrorCode finalize() override;
  PetscErrorCode diagnose(PetscInt t) override;

  static PetscReal get_field(const Vector3R& f);
  static PetscReal get_kinetic(const Vector3R& p, PetscReal m, PetscReal mpw);

protected:
  PetscErrorCode calculate_field();
  PetscErrorCode calculate_kinetic();

  PetscErrorCode calculate_field_change();

  virtual PetscErrorCode fill_energy(PetscInt t);
  virtual PetscErrorCode fill_energy_cons(PetscInt t);

  const interfaces::Simulation& simulation;
  TableDiagnostic energy;
  TableDiagnostic energy_cons;

  PetscReal E = 0, dE = 0, std_E = 0;
  PetscReal B = 0, dB = 0, std_B = 0;
  PetscReal B_total = 0, dB_total = 0;  // Full magnetic-field energy and change.
  PetscReal dF = 0, dK = 0;  // Balance terms corrected for sources and sinks.

  std::vector<PetscReal> K, K0, std_K;

private:
  Vec E_prev = nullptr;
  Vec B_prev = nullptr;
};

#endif  // SRC_DIAGNOSTICS_ENERGY_H
