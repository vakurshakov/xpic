#ifndef SRC_IMPLS_DRIFT_KINETIC_DIAGNOSTIC_H
#define SRC_IMPLS_DRIFT_KINETIC_DIAGNOSTIC_H

#include "src/diagnostics/distribution_moment.h"
#include "src/diagnostics/table_diagnostic.h"
#include "src/impls/drift_kinetic/particles.h"

namespace drift_kinetic {

class Simulation;

using DkMoment = std::vector<PetscReal> (*)(
  const Particles&, const PointByField&, PetscReal lenB);

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

struct DkPhaseAxis {
  PetscReal min = 0.0;
  PetscReal max = 0.0;
  PetscInt bins = 0;
};

/**
 * @brief Five-dimensional histogram of the drift-kinetic distribution in
 * (x, y, z, v_parallel, mu_p). Spatial ownership follows the particle DM;
 */
class DkDistributionFunction : public FieldView {
public:
  static std::unique_ptr<DkDistributionFunction> create(
    const std::string& out_dir, const Particles& particles,
    const DkPhaseAxis& v_parallel, const DkPhaseAxis& mu_p,
    PetscInt diagnose_period, PetscInt max_frames = -1);

  PetscErrorCode diagnose(PetscInt t) override;

private:
  DkDistributionFunction(const std::string& out_dir,
    const Particles& particles, const DkPhaseAxis& v_parallel,
    const DkPhaseAxis& mu_p, PetscInt diagnose_period,
    PetscInt max_frames, MPI_Comm comm);

  PetscErrorCode set_data_views(const Region& region) override;

  const Particles& particles;
  DkPhaseAxis v_parallel;
  DkPhaseAxis mu_p;
  PetscInt max_frames;
  PetscInt frames_written = 0;
};

class EnergyConservation : public TableDiagnostic {
public:
  EnergyConservation(const Simulation& simulation);
  PetscErrorCode diagnose(PetscInt t) override;
  PetscErrorCode initialize() override;
  PetscErrorCode finalize() override;
  PetscErrorCode add_columns(PetscInt t) override;

private:
  PetscErrorCode init_charge_conservation();
  PetscErrorCode collect_charge_density(PetscInt sort_id);
  void calculate_kinetic_energies(std::vector<PetscReal>& per_sort, PetscReal& total) const;

  const Simulation& simulation;
  PetscReal K0 = 0;
  PetscReal K = 0;
  bool initialized = false;
  DM charge_da = nullptr;
  Mat divE = nullptr;
  Vec E_prev = nullptr;
  Vec B_prev = nullptr;
  std::vector<PetscReal> K0_by_sort;
  std::vector<PetscReal> K_by_sort;
  std::vector<Vec> charge_locals;
  std::vector<Vec> charge_fields;
};

} // namespace drift_kinetic

#endif // SRC_IMPLS_DRIFT_KINETIC_DIAGNOSTIC_H
