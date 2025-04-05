#ifndef SRC_ECSIMCORR_PARTICLES_H
#define SRC_ECSIMCORR_PARTICLES_H

#include "src/impls/ecsim/particles.h"

namespace ecsimcorr {

class Simulation;

class Particles : public ecsim::Particles {
public:
  DEFAULT_MOVABLE(Particles);

  Particles(Simulation& simulation, const SortParameters& parameters);
  ~Particles() override;

  PetscErrorCode calculate_energy();

  PetscErrorCode clear_sources() override;
  PetscErrorCode first_push() override;
  PetscErrorCode second_push() override;
  PetscErrorCode final_update();

  using ecsim::Particles::E;
  using ecsim::Particles::B;

  using ecsim::Particles::local_currI;
  using ecsim::Particles::global_currI;
  using ecsim::Particles::currI;

  Vec local_currJe;
  Vec global_currJe;
  Vector3R*** currJe;

private:
  static constexpr auto& shape_func2 = spline_of_2nd_order;
  static constexpr PetscReal shape_radius2 = 1.5;

  void decompose_esirkepov_current(const Shape& shape, const Point& point);

  Simulation& simulation_;

  PetscReal energy = 0.0;
  PetscReal pred_w = 0.0;
  PetscReal corr_w = 0.0;
  PetscReal pred_dK = 0.0;
  PetscReal corr_dK = 0.0;
  PetscReal lambda_dK = 0.0;

  PetscClassId classid;
  PetscLogEvent events[3];

  friend class EnergyConservation;
};

}  // namespace ecsimcorr

#endif  // SRC_ECSIMCORR_PARTICLES_H
