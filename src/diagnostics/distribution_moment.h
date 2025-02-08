#ifndef SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
#define SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H

#include "src/interfaces/particles.h"
#include "src/diagnostics/field_view.h"

using Moment = PetscReal (*)(const interfaces::Particles&, const Point&);

/// @note To see the list of available moment getters, check the implementation
Moment moment_from_string(const std::string& name);


namespace ecsimcorr {
class ChargeConservation;
}

/**
 * @brief Diagnostic of particles _coordinate_ distribution moment
 * @todo Collect more possible diagnostics (components) at once as for FieldView (?)
 */
class DistributionMoment : public FieldView {
public:
  DEFAULT_MOVABLE(DistributionMoment);

  static std::unique_ptr<DistributionMoment> create(const std::string& out_dir,
    const interfaces::Particles& particles, const Moment& moment,
    const Region& region);

  ~DistributionMoment() override;

  PetscErrorCode diagnose(timestep_t t) override;

private:
  DistributionMoment(const std::string& out_dir,
    const interfaces::Particles& particles, const Moment& moment,
    MPI_Comm newcomm);

  PetscErrorCode set_data_views(const Region& region);
  PetscErrorCode set_da(const Region& region);

  PetscErrorCode collect();

  Vec local_;

  const interfaces::Particles& particles_;
  Moment moment_;

  friend class ecsimcorr::ChargeConservation;
};

#endif  // SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
