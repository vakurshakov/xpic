#ifndef SRC_IMPLS_HYBRID_KINETIC_DIAGNOSTIC_H
#define SRC_IMPLS_HYBRID_KINETIC_DIAGNOSTIC_H

#include "src/interfaces/particles.h"
#include "src/diagnostics/table_diagnostic.h"

namespace hybrid_kinetic {

class Simulation;

/// @brief Total energy balance and per-step continuity residual for both
/// kinetic and drift-kinetic sorts, columns follow `dk_diagnostic.txt`.
class EnergyConservation : public TableDiagnostic {
public:
  EnergyConservation(const Simulation& simulation);
  PetscErrorCode initialize() override;
  PetscErrorCode finalize() override;
  PetscErrorCode add_columns(PetscInt t) override;

private:
  struct SortView {
    const interfaces::Particles& sort;
    std::function<PetscReal()> kinetic_energy_local;
    std::function<PetscErrorCode(Vec)> collect_charge_density;
  };

  PetscErrorCode calculate_kinetic_energies(std::vector<PetscReal>& K) const;

  const Simulation& simulation;
  std::vector<SortView> sorts;  // Drift-kinetic sorts first, then kinetic ones.
  bool initialized = false;

  PetscReal w_E0 = 0;         // Electric field energy at the previous step.
  PetscReal w_B0 = 0;         // Magnetic field energy at the previous step.
  std::vector<PetscReal> K0;  // Kinetic energies at the previous step.
  std::vector<Vec> rho0;      // Charge densities at the previous step.
};

}  // namespace hybrid_kinetic

#endif  // SRC_IMPLS_HYBRID_KINETIC_DIAGNOSTIC_H
