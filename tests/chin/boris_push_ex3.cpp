#include "common.h"

// clang-format off
static char help[] =
  "Recreation of published results, see https://doi.org/10.1016/j.jcp.2022.111422    \n"
  "Here we are testing the electron drift in a curvilinear magnetic field created    \n"
  "by the line current flowing along z-axis. It is described by B_coeff and B_center \n"
  "parameters. Different process algorithms are tested. None that only \"B\" schemes \n"
  "can be used since `Omega * dt >> 1.0`.\n";
// clang-format on

constexpr PetscReal B_coeff = 800;
constexpr Vector3R B_center(10, 10, 0);

InterpolationResult get_magnetic_field(const Vector3R& r);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  std::string chin_scheme_id;
  PetscCall(get_id(chin_scheme_id));

  constexpr Vector3R r0(0, 10, 0);
  constexpr Vector3R v0(0.16, 1, 0);

  Point point{r0, v0};
  auto particles = prepare_electron(point);

  dt = 0.16;
  geom_nt = 1000;

  Vector3R check_mean_v;

  SyncFile output(get_outputfile(__FILE__, chin_scheme_id));
  output() << "t       x       y       z       \n";
  output() << "[1/wpe] [c/wpe] [c/wpe] [c/wpe] \n";

  BorisPush push;

  if (chin_scheme_id.ends_with("LF"))
    point.r -= (dt / 2.0) * point.p;

  for (PetscInt t = 0; t < geom_nt; ++t) {
    output() << t * dt << " " << point.r << "\n";
    process_impl(chin_scheme_id, push, point, *particles, get_magnetic_field);

    check_mean_v += point.p / static_cast<PetscReal>(geom_nt);
  }

  Vector3R B_p{0, -B_coeff / r0.length(), 0};
  Vector3R B_grad{B_coeff / r0.squared(), 0, 0};

  /// @note v_{\grad(B)} = mv_{\perp}^2 / (2qB) * [B x \grad(B)] / B^2;
  Vector3R v_drift = particles->mass(point) *
    (POW2(v0.x()) + 2.0 * POW2(v0.y())) /
    (2.0 * particles->charge(point) * B_p.length()) * B_p.cross(B_grad) /
    B_p.squared();

  PetscCheck(check_mean_v.dot(v_drift) > 0.0, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Particle should drift along theoretical prediction, having mean: (%f, %f, %f), theory: (%f, %f, %f)", REP3_A(check_mean_v), REP3_A(v_drift));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

InterpolationResult get_magnetic_field(const Vector3R& r)
{
  Vector3R cr = r - B_center;
  PetscReal rr = cr.length();
  PetscReal ra = std::atan2(cr.y(), cr.x());

  PetscReal B_theta = B_coeff / rr;

  return std::make_pair(Vector3R{},
    Vector3R{
      -std::sin(ra) * B_theta,
      +std::cos(ra) * B_theta,
      0.0,
    });
}
