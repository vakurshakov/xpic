#ifndef SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
#define SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H

#include "src/interfaces/particles.h"
#include "src/diagnostics/field_view.h"

/// @note The size of the returned `vector` should match the passed `region.dof`
using Moment = std::vector<PetscReal> (*)(
  const interfaces::Particles&, const Point&);

/// @note To see the list of available moment getters, check the implementation
Moment moment_from_string(const std::string& name);


/// @brief Diagnostic of particles _coordinate_ distribution moment
class DistributionMoment : public FieldView {
  friend class ChargeConservation;

public:
  DEFAULT_MOVABLE(DistributionMoment);

  static std::unique_ptr<DistributionMoment> create(const std::string& out_dir,
    const interfaces::Particles& particles, const Moment& moment,
    const Region& region);

  PetscErrorCode finalize() override;
  PetscErrorCode diagnose(PetscInt t) override;

protected:
  DistributionMoment(const interfaces::Particles& particles);

  DistributionMoment(const std::string& out_dir,
    const interfaces::Particles& particles, const Moment& moment,
    MPI_Comm newcomm);

  PetscErrorCode set_data_views(const Region& region) override;
  PetscErrorCode set_local_da(const Region& region);

  struct Shape;
  virtual PetscErrorCode collect();

  DM global_da_;
  Vec local_;

  const interfaces::Particles& particles_;
  Moment moment_;
};

#endif  // SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
