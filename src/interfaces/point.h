#ifndef SRC_INTERFACES_POINT_H
#define SRC_INTERFACES_POINT_H

#include "src/utils/vector3.h"

struct Point {
  Vector3R r;
  Vector3R p;

  // clang-format off: access modifiers
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
  // clang-format on
};

struct PointByField {
  Vector3R r;
  Vector3R p;
  PetscReal p_parallel;
  PetscReal mu_p;

  /// @brief `point.r` is already the guiding center.
  static PointByField from_guiding_center(
    const Point& point, const Vector3R& Bp, PetscReal mp)
  {
    const Vector3R b = Bp.normalized();
    const PetscReal lenB = Bp.length();
    const PetscReal p_perp = point.p.transverse_to(Bp).length();

    return PointByField{
      .r = point.r,
      .p = point.p,
      .p_parallel = point.p.dot(b),
      .mu_p = mp * p_perp * p_perp / (2.0 * lenB),
    };
  }

  /// @brief `point.r` is the particle position, shifted by the Larmor vector.
  static PointByField from_particle_position(
    const Point& point, const Vector3R& Bp, PetscReal mp, PetscReal qm)
  {
    Point gc{
      .r = point.r + point.p.cross(Bp.normalized()) / (qm * Bp.length()),
      .p = point.p,
    };
    return from_guiding_center(gc, Bp, mp);
  }

  // clang-format off: access modifiers
  PetscReal& x() { return r.x(); }
  PetscReal& y() { return r.y(); }
  PetscReal& z() { return r.z(); }

  PetscReal x() const { return r.x(); }
  PetscReal y() const { return r.y(); }
  PetscReal z() const { return r.z(); }

  PetscReal& p_par() { return p_parallel; }
  PetscReal p_par() const { return p_parallel; }

  PetscReal& mu() { return mu_p; }
  PetscReal mu() const { return mu_p; }
  // clang-format off: access modifiers

  /**
   * @brief Perpendicular momentum recovered from the stored momentum vector:
   * $p_\perp = \sqrt{|p|^2 - p_\parallel^2}$. Equivalent to the adiabatic
   * form $\sqrt{2\mu B / m}$ at construction (`mu_p` is the conserved
   * invariant kept instead of a redundant `p_perp` member).
   */
  PetscReal p_perp() const
  {
    return sqrt(std::max(0.0, p.squared() - p_parallel * p_parallel));
  }
};

void g_bound_reflective(Point& point, Axis axis);
void g_bound_periodic(Point& point, Axis axis);
void g_bound_periodic(PointByField& point, Axis axis);

#endif  // SRC_INTERFACES_POINT_H
