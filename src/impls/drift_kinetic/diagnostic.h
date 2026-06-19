#ifndef SRC_IMPLS_DRIFT_KINETIC_DIAGNOSTIC_H
#define SRC_IMPLS_DRIFT_KINETIC_DIAGNOSTIC_H

#include "src/diagnostics/distribution_moment.h"
#include "src/diagnostics/table_diagnostic.h"
#include "src/impls/drift_kinetic/particles.h"
#include "src/utils/configuration.h"

namespace drift_kinetic {

class Simulation;

/// @brief Moment getter taking a drift-kinetic `PointByField` directly so it
/// can read `p_parallel`, `p_perp`, `mu_p` instead of going through the
/// gyrocenter velocity stored in `Point::p`. `lenB` is the magnitude of the
/// current magnetic field interpolated to the gyrocenter (used by
/// `temperature_perp`, ignored by the others).
using DkMoment = std::vector<PetscReal> (*)(
  const Particles&, const PointByField&, PetscReal lenB);

/// @brief Map a moment name to a DK-specific getter. Returns nullptr if the
/// name is not a DK-specific moment (callers should then fall back to the
/// shared `moment_from_string`).
DkMoment dk_moment_from_string(const std::string& name);

class DkDistributionMoment : public ::DistributionMoment {
public:
  static std::unique_ptr<DkDistributionMoment> create(const std::string& out_dir,
    const Particles& particles, const Moment& moment, const Region& region);
  static std::unique_ptr<DkDistributionMoment> create(const std::string& out_dir,
    const Particles& particles, const DkMoment& moment, const Region& region);

protected:
  DkDistributionMoment(const std::string& out_dir,
    const Particles& particles, const Moment& moment, MPI_Comm newcomm);
  DkDistributionMoment(const std::string& out_dir,
    const Particles& particles, const DkMoment& moment, MPI_Comm newcomm);

  PetscErrorCode collect() override;

  const Particles& dk_particles;
  DkMoment dk_moment = nullptr;
};

/// @brief FieldView wrapper that applies a linear operator (a `Mat`) to a
/// source `Vec` at every diagnose() call and writes the result via the
/// inherited FieldView machinery. Generic enough to cover any
/// rotor-of-X / Mat-times-Vec diagnostic — the builder picks the
/// operator and source depending on the field name (rotE / rotB / rotM /
/// <sort>/rotM, ...) without forcing the simulation to keep extra Vecs.
class MatMultFieldView : public ::FieldView {
public:
  static std::unique_ptr<MatMultFieldView> create(const std::string& out_dir,
    DM da, Vec source, Mat op, const Region& region);

  PetscErrorCode finalize() override;
  PetscErrorCode diagnose(PetscInt t) override;

protected:
  MatMultFieldView(const std::string& out_dir, DM da, Vec field,
    Vec source, Mat op, MPI_Comm newcomm);

  Vec source = nullptr;
  Mat op = nullptr;
};

class PointByFieldTrace : public TableDiagnostic {
public:
  PointByFieldTrace(const std::string& out_dir, const Particles& particles, PetscInt skip = 1);


  PetscErrorCode initialize() override;
  PetscErrorCode finalize() override;
  PetscErrorCode diagnose(PetscInt t) override;

private:
  PetscInt skip;
  const Particles& particles;

  PetscErrorCode add_columns(PetscInt t) override;
};

class EnergyConservation : public TableDiagnostic {
public:
  EnergyConservation(const Simulation& simulation);
  PetscErrorCode diagnose(PetscInt t) override;
  PetscErrorCode initialize() override;
  PetscErrorCode finalize() override;
  PetscErrorCode add_columns(PetscInt t) override;

  const Simulation& simulation;
  PetscReal w_E = 0, w_E0 = 0;
  PetscReal w_B = 0, w_B0 = 0;
  PetscReal dF = 0;
  PetscReal a_EJ = 0;
  PetscReal a_MB = 0, a_MB0 = 0;
  PetscReal w_M = 0, w_Mn = 0;
  PetscReal K0 = 0, K = 0;
  bool initialized = false;

private:
  PetscErrorCode init_charge_conservation();
  PetscErrorCode collect_charge_density(PetscInt sort_id);
  PetscErrorCode collect_charge_densities();
  void calculate_kinetic_energies(std::vector<PetscReal>& per_sort, PetscReal& total) const;

  DM charge_da = nullptr;
  Mat divE = nullptr;
  std::vector<PetscReal> K0_by_sort;
  std::vector<PetscReal> K_by_sort;
  std::vector<Vec> charge_locals;
  std::vector<Vec> charge_fields;
  std::vector<Vec> current_densities;
};

} // namespace drift_kinetic

#endif // SRC_IMPLS_DRIFT_KINETIC_DIAGNOSTIC_H
