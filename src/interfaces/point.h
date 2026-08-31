#ifndef SRC_INTERFACES_POINT_H
#define SRC_INTERFACES_POINT_H

#include "src/pch.h"
#include "src/utils/vector3.h"

struct Point {
  Vector3R r;
  Vector3R p;

  constexpr Point() = default;

  constexpr Point(const Vector3R& r, const Vector3R& p)
    : r(r), p(p)
  {
  }

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

  PointByField() = default;

  PointByField(const Vector3R& r, PetscReal p_parallel, PetscReal mu_p)
    : r(r), p{}, p_parallel(p_parallel), mu_p(mu_p)
  {}

  PointByField(const Point& point, const Vector3R& Bp, PetscReal mp, PetscReal qm)
    : r(point.r + point.p.cross(Bp.normalized()) / (qm * Bp.length())),
      p(point.p),
      p_parallel(point.p.dot(Bp.normalized())),
      mu_p(mp * point.p.transverse_to(Bp).squared() / (2.0 * Bp.length()))
  {}

  // clang-format off: access modifiers
  PetscReal& x() { return r.x(); }
  PetscReal& y() { return r.y(); }
  PetscReal& z() { return r.z(); }

  PetscReal x() const { return r.x(); }
  PetscReal y() const { return r.y(); }
  PetscReal z() const { return r.z(); }

  PetscReal& p_par() { return p_parallel; }
  PetscReal p_par() const { return p_parallel; }

  /// @brief Perpendicular momentum recovered from the stored momentum vector:
  /// @f$p_\perp = \sqrt{|p|^2 - p_\parallel^2}@f$. Equivalent to the adiabatic
  /// form @f$\sqrt{2\mu B / m}@f$ at construction (`mu_p` is the conserved
  /// invariant kept instead of a redundant `p_perp` member).
  PetscReal p_perp() const
  {
    return std::sqrt(std::max(0.0, p.squared() - p_parallel * p_parallel));
  }

  PetscReal& mu() { return mu_p; }
  PetscReal mu() const { return mu_p; }
  // clang-format off: access modifiers
};

void g_bound_reflective(Point& point, Axis axis);
void g_bound_periodic(Point& point, Axis axis);
void g_bound_periodic(PointByField& point, Axis axis);

#endif  // SRC_INTERFACES_POINT_H
