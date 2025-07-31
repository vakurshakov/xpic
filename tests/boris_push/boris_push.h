#include "src/algorithms/boris_push.h"
#include "src/utils/vector3.h"
#include "tests/common.h"

PetscErrorCode get_id(std::string& id)
{
  PetscFunctionBeginUser;
  PetscBool flg;
  char id_c_str[5];
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-id", id_c_str, 5, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must specify the Chin scheme id with '-id' option");
  id = std::move(id_c_str);
  PetscFunctionReturn(PETSC_SUCCESS);
}

using Interpolator = std::function<void(const Vector3R&, Vector3R&, Vector3R&)>;

// First-order magnetic field integrators

void process_M1A(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  Vector3R E_p, B_p;
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vM(dt, point);
  push.update_r(dt, point);
}

void process_M1B(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  push.update_r(dt, point);

  Vector3R E_p, B_p;
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vM(dt, point);
}

void process_MLF(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  process_M1B(push, point, interpolate);
}

void process_B1A(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  Vector3R E_p, B_p;
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vB(dt, point);
  push.update_r(dt, point);
}

void process_B1B(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  push.update_r(dt, point);

  Vector3R E_p, B_p;
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vB(dt, point);
}

void process_BLF(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  process_B1B(push, point, interpolate);
}

void process_C1A(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  Vector3R E_p, B_p;
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vC1(dt, point);
  push.update_r(dt, point);
}

void process_C1B(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  push.update_r(dt, point);

  Vector3R E_p, B_p;
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vC1(dt, point);
}

void process_CLF(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  process_C1B(push, point, interpolate);
}


// Second-order magnetic field integrators

void process_M2A(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  // v_B(r_0, v_0, dt / 2) -> v_{1/2}
  Vector3R E_p, B_p;
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vM(dt / 2.0, point);

  // r_0 + dt * v_{1/2} -> r_1
  push.update_r(dt, point);

  // v_B(r_1, v_{1/2}, dt / 2) ;-> v_1
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vM(dt / 2.0, point);
}

void process_M2B(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  // r_0 + (dt / 2) * v_0 -> r_{1/2}
  push.update_r(dt / 2.0, point);

  // v_B(r_{1/2}, v_0, dt) -> v_1
  Vector3R E_p, B_p;
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vM(dt, point);

  // r_{1/2} + (dt / 2) * v_1 -> r_1
  push.update_r(dt / 2.0, point);
}

void process_C2A(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  // v_B(r_0, v_0, dt / 2) -> v_{1/2}
  Vector3R E_p, B_p;
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vC2(dt / 2.0, point);

  // r_0 + dt * v_{1/2} -> r_1
  push.update_r(dt, point);

  // v_B(r_1, v_{1/2}, dt / 2) ;-> v_1
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vC2(dt / 2.0, point);
}

void process_B2B(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  // r_0 + (dt / 2) * v_0 -> r_{1/2}
  push.update_r(dt / 2.0, point);

  // v_B(r_{1/2}, v_0, dt) -> v_1
  Vector3R E_p, B_p;
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vB(dt, point);

  // r_{1/2} + (dt / 2) * v_1 -> r_1
  push.update_r(dt / 2.0, point);
}

// Electro-magnetic field integrators

void process_EB1A(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  Vector3R E_p, B_p;
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vEB(dt, point);
  push.update_r(dt, point);
}

void process_EB1B(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  push.update_r(dt, point);

  Vector3R E_p, B_p;
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vEB(dt, point);
}

void process_EBLF(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  process_EB1B(push, point, interpolate);
}

void process_EB2B(BorisPush& push, Point& point, const Interpolator& interpolate)
{
  // r_0 + (dt / 2) * v_0 -> r_{1/2}
  push.update_r(dt / 2.0, point);

  // v_B(r_{1/2}, v_0, dt) -> v_1
  Vector3R E_p, B_p;
  interpolate(point.r, E_p, B_p);
  push.set_fields(E_p, B_p);
  push.update_vEB(dt, point);

  // r_{1/2} + (dt / 2) * v_1 -> r_1
  push.update_r(dt / 2.0, point);
}


void process_impl(std::string_view id, BorisPush& push, Point& point,
  const Interpolator& interpolate)
{
  using process_func =
    void (*)(BorisPush& push, Point& point, const Interpolator& interpolate);

  static const std::map<std::string_view, process_func> map{
    {"M1A", process_M1A},
    {"M1B", process_M1B},
    {"MLF", process_MLF},
    {"B1A", process_B1A},
    {"B1B", process_B1B},
    {"BLF", process_BLF},
    {"C1A", process_C1A},
    {"C1B", process_C1B},
    {"CLF", process_CLF},
    {"M2A", process_M2A},
    {"M2B", process_M2B},
    {"C2A", process_C2A},
    {"B2B", process_B2B},
    {"EB1A", process_EB1A},
    {"EB1B", process_EB1B},
    {"EBLF", process_EBLF},
    {"EB2B", process_EB2B},
  };

  (*map.at(id))(push, point, interpolate);
}


PetscReal get_effective_larmor(std::string_view id, PetscReal rg, PetscReal theta)
{
  if (id.starts_with("M") && id != "M2B")
    return rg * (theta / 2.0) / std::sin(theta / 2.0);
  if (id.starts_with("B") && id != "B2B")
    return rg * std::sqrt(1.0 + POW2(theta) / 4.0);
  if (id == "M2B")
    return rg * (theta / 2.0) / std::tan(theta / 2.0);
  return rg;
}


Vector3R get_center_offset(std::string_view id, PetscReal rg, PetscReal theta)
{
  Vector3R rc;

  if (id.starts_with("M")) {
    PetscReal Rg = get_effective_larmor(id, rg, theta);

    if (id.starts_with("M1") || id == "MLF") {
      rc[X] = rg - Rg * std::cos(theta / 2.0);
      rc[Y] = rg * theta / 2.0;
    }
    else if (id.starts_with("M2")) {
      rc[X] = rg - Rg;
      return rc;
    }
  }

  if (id.starts_with("B")) {
    if (id == "B2B")
      return rc;

    PetscReal sin_tb = theta /*               */ / (1.0 + POW2(theta) / 4.0);
    PetscReal cos_tb = (1.0 - POW2(theta) / 4.0) / (1.0 + POW2(theta) / 4.0);
    rc[X] = 0.0;
    rc[Y] = rg * (1 - cos_tb) / sin_tb;  // tan(theta_b / 2) = theta / 2;
  }

  if (id.starts_with("C")) {
    if (id == "C2A")
      return rc;

    PetscReal cos_tc = (1.0 - POW2(theta) / 2.0);
    rc[X] = rg * (1.0 - std::sqrt(1.0 - POW2(theta) / 4.0));
    rc[Y] = rg * std::sqrt((1.0 - cos_tc) / 2.0);
  }

  if (id.ends_with("A"))
    rc[Y] *= (-1.0);
  else if (id.ends_with("LF"))
    rc[Y] = 0.0;

  return rc;
}
