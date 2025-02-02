#ifndef SRC_ECSIMCORR_PARTICLES_H
#define SRC_ECSIMCORR_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/utils/shape.h"

namespace ecsimcorr {

class Simulation;

class Particles : public interfaces::Particles {
public:
  DEFAULT_MOVABLE(Particles);

  Particles(Simulation& simulation, const SortParameters& parameters);
  ~Particles() override;

  PetscErrorCode calculate_energy();
  PetscErrorCode clear_sources();
  PetscErrorCode first_push();
  PetscErrorCode second_push();
  PetscErrorCode final_update();

  PetscErrorCode fill_ecsim_current(
    MatStencil* coo_i, MatStencil* coo_j, PetscReal* coo_v);

private:
  static constexpr PetscInt OMP_CHUNK_SIZE = 16;

#if (PARTICLES_FORM_FACTOR == 2)
  static constexpr auto& shape_func1 = spline_of_1st_order;
  static constexpr auto& shape_func2 = spline_of_2nd_order;
  static constexpr PetscReal shape_radius1 = 1.0;
  static constexpr PetscReal shape_radius2 = 1.5;
#endif

  void decompose_esirkepov_current(const Shape& shape, const Point& point);

  void decompose_ecsim_current(const Shape& shape, const Point& point,
    const Vector3R& B_p, PetscReal* coo_v);

  void fill_matrix_indices(PetscInt g, MatStencil* coo_i, MatStencil* coo_j);

  constexpr PetscInt ind(PetscInt g, PetscInt c1, PetscInt c2)
  {
    return g * POW2(3) + (c1 * 3 + c2);
  }

  Simulation& simulation_;

  Vec local_currI;
  Vec local_currJe;
  Vec global_currI;
  Vec global_currJe;
  Vector3R*** currI;
  Vector3R*** currJe;

  PetscReal energy = 0.0;
  PetscReal pred_w = 0.0;
  PetscReal corr_w = 0.0;
  PetscReal pred_dK = 0.0;
  PetscReal corr_dK = 0.0;
  PetscReal lambda_dK = 0.0;

  PetscClassId classid;
  PetscLogEvent events[3];

  friend class EnergyConservation;
  friend class ChargeConservation;
};

}  // namespace ecsimcorr

#endif  // SRC_ECSIMCORR_PARTICLES_H
