#ifndef SRC_INTERFACES_POINT_H
#define SRC_INTERFACES_POINT_H

#include "src/pch.h"
#include "src/utils/vector3.h"

//namespace utils_point {
//  Vector3R R_correction(const Vector3R& vp, const Vector3R& Bp, PetscReal qm) {
//    return vp.cross(Bp.normalized())/(qm*Bp.length());
//  }
//}

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
  PetscReal p_parallel;
  /// @todo del p_perp or mu_p
  PetscReal p_perp;
  PetscReal mu_p;

  PointByField() = default;

  PointByField(
    const Vector3R& r, PetscReal p_perp, PetscReal p_parallel, PetscReal mu_p)
    : r(r), p_parallel(p_parallel), p_perp(p_perp), mu_p(mu_p)
  {
  }

  PointByField(const Point& point, const Vector3R& Bp, PetscReal mp, PetscReal qm)
    : r(point.r - point.p.cross(Bp.normalized())/(qm*Bp.length())),
      p_parallel(point.p.parallel_to(Bp).length()),
      p_perp(point.p.transverse_to(Bp).length()),
      mu_p(mp * p_perp * p_perp / (2.0 * Bp.length()))
  {
  }

  // clang-format off: access modifiers
  PetscReal& x() { return r.x(); }
  PetscReal& y() { return r.y(); }
  PetscReal& z() { return r.z(); }

  PetscReal x() const { return r.x(); }
  PetscReal y() const { return r.y(); }
  PetscReal z() const { return r.z(); }

  PetscReal& p_par() { return p_parallel; }
  PetscReal& p_perp_ref() { return p_perp; }
  PetscReal p_par() const { return p_parallel; }
  PetscReal p_perp_ref() const { return p_perp; }

  PetscReal& mu() { return mu_p; }
  PetscReal mu() const { return mu_p; }
  // clang-format off: access modifiers
};

void g_bound_reflective(Point& point, Axis axis);
void g_bound_periodic(Point& point, Axis axis);

#endif  // SRC_INTERFACES_POINT_H
