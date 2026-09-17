
#include "src/interfaces/particles.h"
#include "src/algorithms/boris_push.h"
#include "src/algorithms/crank_nicolson_push.h"
#include "src/algorithms/drift_kinetic_push.h"
#include "src/utils/utils.h"
#include "src/utils/vector3.h"
#include "src/utils/world.h"
#include "tests/common.h"
#include "src/utils/configuration.h"

#include <string>
#include <string_view>
#include <vector>

constexpr PetscReal q = -1.0;
constexpr PetscReal m = +1.0;

template<typename GetFields>
void boris_step(BorisPush& push, Point& point, GetFields get_fields)
{
  push.update_r(0.5 * dt, point);

  Vector3R E_p, B_p, stub_gradB;
  get_fields(point.r, point.r, E_p, B_p, stub_gradB);
  push.set_fields(E_p, B_p);
  push.update_vEB(dt, point);

  push.update_r(0.5 * dt, point);
}


namespace gaussian_magnetic_mirror {

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 2.0;
constexpr PetscReal L = 10.0;      // Half the length of the trap
constexpr PetscReal W = 1.0;      // Mirror width
constexpr PetscReal S = POW2(W);  // Mirror width squared
constexpr PetscReal Rc = 5.0;

PetscReal exp(PetscReal z, PetscReal z0)
{
  return std::exp(-POW2(z - z0) / S);
}

// Center field profile on the axis (double gauss)
PetscReal get_Bz(PetscReal z)
{
  return B_min + (B_max - B_min) * (exp(z, -L) + exp(z, +L));
}

PetscReal get_dBz_dz(PetscReal z)
{
  return (B_max - B_min) * //
    ((-2.0 * (z + L) / S * exp(z, -L)) + //
      (-2.0 * (z - L) / S * exp(z, +L)));
}

PetscReal get_d2Bz_dz2(PetscReal z)
{
  PetscReal t1 = (z + L);
  PetscReal t2 = (z - L);
  return (B_max - B_min) * //
    ((-2.0 / S + 4.0 * POW2(t1 / S)) * exp(z, -L) + //
      (-2.0 / S + 4.0 * POW2(t2 / S)) * exp(z, +L));
}

PetscReal get_d3Bz_dz3(PetscReal z)
{
  PetscReal t1 = (z + L);
  PetscReal t2 = (z - L);
  return (B_max - B_min) * //
    ((12.0 * t1 / (S * S) - 8.0 * POW3(t1 / S)) * exp(z, -L) + //
      (12.0 * t2 / (S * S) - 8.0 * POW3(t2 / S)) * exp(z, +L));
}

// Bz field off axis
PetscReal get_Bz_corr(const Vector3R& r)
{
  return get_Bz(r.z() - L) -
    0.25 * (POW2(r.x() - Rc) + POW2(r.y() - Rc)) * get_d2Bz_dz2(r.z() - L);
}

void get_fields(const Vector3R&, const Vector3R& pos, //
  Vector3R&, Vector3R& B_p, Vector3R& gradB_p)
{
  PetscReal x = pos.x() - Rc;
  PetscReal y = pos.y() - Rc;
  PetscReal z = pos.z() - L;
  PetscReal r2 = x * x + y * y;
  PetscReal r = std::sqrt(r2);

  // 1) Calculation of the axial field and its derivatives
  PetscReal Bz = get_Bz(z);
  PetscReal dBz_dz = get_dBz_dz(z);
  PetscReal d2Bz_dz2 = get_d2Bz_dz2(z);
  PetscReal d3Bz_dz3 = get_d3Bz_dz3(z);

  // 2) Computation of the magnetic field vector B_p in the paraxial
  // approximation: transverse components, follow from div(B) = 0,
  // longitudinal component with second-order correction by radius.
  B_p = Vector3R{
    -0.5 * x * dBz_dz,
    -0.5 * y * dBz_dz,
    Bz - 0.25 * r2 * d2Bz_dz2,
  };

  // 3) Calculation of the field modulus gradient |B| from full B vector.
  Vector3R dBdx{
    -0.5 * dBz_dz,
    0.0,
    -0.5 * x * d2Bz_dz2,
  };
  Vector3R dBdy{
    0.0,
    -0.5 * dBz_dz,
    -0.5 * y * d2Bz_dz2,
  };
  Vector3R dBdz{
    -0.5 * x * d2Bz_dz2,
    -0.5 * y * d2Bz_dz2,
    dBz_dz - 0.25 * r2 * d3Bz_dz3,
  };

  PetscReal B_len = B_p.length();
  if (B_len > 1e-12) {
    gradB_p = Vector3R{
      B_p.dot(dBdx),
      B_p.dot(dBdy),
      B_p.dot(dBdz),
    } / B_len;
  }
  else {
    gradB_p = {};
  }
}

void get_grid(const Vector3R&, const Vector3R& pos, //
  Vector3R&, Vector3R& B_p, Vector3R& gradB_p)
{
  Vector3R E_dummy;
  Vector3R Bx, By, Bz;
  Vector3R gBx, gBy, gBz;

  get_fields({}, {pos.x(), pos.y() + 0.5 * dy, pos.z() + 0.5 * dz}, E_dummy, Bx, gBx);
  get_fields({}, {pos.x() + 0.5 * dx, pos.y(), pos.z() + 0.5 * dz}, E_dummy, By, gBy);
  get_fields({}, {pos.x() + 0.5 * dx, pos.y() + 0.5 * dy, pos.z()}, E_dummy, Bz, gBz);

  B_p = {Bx.x(), By.y(), Bz.z()};
  gradB_p = {gBx.x(), gBy.y(), gBz.z()};
}

} // namespace gaussian_magnetic_mirror
