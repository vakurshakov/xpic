#ifndef SRC_INTERFACES_PARTICLES_POINT_H
#define SRC_INTERFACES_PARTICLES_POINT_H

#include "src/pch.h"
#include "src/vectors/vector3.h"
#include "src/interfaces/particles/parameters.h"

namespace interfaces { class Particles; }

/**
 * @brief Describes the point in 6D space of coordinates and momentum.
 *
 * @details
 * The above describes the common use case. It can be a higher order
 * point in case of compilation with, for example, per-particle density
 * storage. However, particle related parameters such as density, charge
 * and mass are available only through `interfaces::Particles` class.
 *
 * The intention behind such separation is the following:
 * 1) We want to simplify the "moving parts" of the particles, so we
 *    can easily use standard algorithms without thinking about parameters pointer.
 * 2) This way we reduce the additional cost of a pointer size or size of a
 *    particle tag which would be used to address a global parameters.
 */
class Point {
public:
  Vector3R r = 0.0;
  Vector3R p = 0.0;

  Point() = default;
  Point(
    const Vector3R& r,
    const Vector3R& p
#if PARTICLES_LOCAL_PNUM
    , PetscInt Np
#endif
#if PARTICLES_LOCAL_DENSITY
    , PetscReal n
#endif
  );

  PetscReal& x() { return r.x(); }
  PetscReal& y() { return r.y(); }
  PetscReal& z() { return r.z(); }

  PetscReal x() const { return r.x(); }
  PetscReal y() const { return r.y(); }
  PetscReal z() const { return r.z(); }

  PetscReal& px() { return p.x(); }
  PetscReal& py() { return p.y(); }
  PetscReal& pz() { return p.z(); }

  PetscReal px() const { return p.x(); }
  PetscReal py() const { return p.y(); }
  PetscReal pz() const { return p.z(); }

private:
  friend class interfaces::Particles;

#if PARTICLES_LOCAL_PNUM
  PetscReal __Np;
#endif
#if PARTICLES_LOCAL_DENSITY
  PetscReal __n;
#endif
};

void g_bound_reflective(Point& point, Axis axis);
void g_bound_periodic(Point& point, Axis axis);

#endif  // SRC_INTERFACES_PARTICLES_POINT_H
