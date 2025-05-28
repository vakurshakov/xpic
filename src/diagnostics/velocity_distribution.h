#ifndef SRC_DIAGNOSTICS_VELOCITY_DISTRIBUTION_H
#define SRC_DIAGNOSTICS_VELOCITY_DISTRIBUTION_H

#include "src/diagnostics/distribution_moment.h"

/// @brief Customizable projector of particle velocities
using Projector = std::tuple<PetscReal, PetscReal> (*)(const Point&);

/// @note To see the list of available projectors, check the implementation
Projector projector_from_string(const std::string& name);

/// @brief Default geometry tester from "src/utils/geometries.h"
using Tester = std::function<bool(const Vector3R&)>;

/**
 * @brief This diagnostic collects velocity distribution
 * with some restrictions on space coordinates and velocity.
 *
 * @note Because particles are distributed between different
 * mpi ranks (in general), calculation of the common velocity
 * distribution requires the equivalent of `MPI_Allreduce()`
 * operation. So, firstly partial distributions f(v) are
 * collected on separate ranks and then summed using `VecScatter`.
 */
class VelocityDistribution : public DistributionMoment {
public:
  struct VelocityRegion {
    PetscReal vx_max, vy_max;
    PetscReal vx_min, vy_min;
    PetscReal dvx, dvy;
  };

  static std::unique_ptr<VelocityDistribution> create( //
    const std::string& out_dir, const interfaces::Particles& particles,
    const Projector& projector, const Tester& within_geom,
    const Region& xreg_aabb, const VelocityRegion& vreg);

  PetscErrorCode finalize() override;

protected:
  VelocityDistribution(const std::string& out_dir,
    const interfaces::Particles& particles, const Projector& projector,
    const Tester& within_geom, MPI_Comm newcomm);

  PetscErrorCode set_regions(const Region& xreg_aabb, const VelocityRegion& vreg);
  PetscErrorCode set_data_views(const Region& /* reg */) override;

  PetscErrorCode collect() override;

  Region xreg_aabb;
  VelocityRegion vreg;

  Vector3I xstart, xsize;
  Vector3I vstart, vsize;

  Projector projector;
  Tester within_geom;

  IS is;
  VecScatter ctx;
};

#endif  // SRC_DIAGNOSTICS_VELOCITY_DISTRIBUTION_H
