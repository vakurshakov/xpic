#include "common.h"

static char help[] =
  "Here we are testing the electron drift in a curvilinear magnetic \n"
  "field created by the line current flowing along z-axis. It is    \n"
  "described by B_coeff and B_center parameters. Different process  \n"
  "algorithms are tested. None that only \"B\" schemes can be used, \n"
  "since `Omega * dt >> 1.0`.  This is a recreation of published    \n"
  "results, see https://doi.org/10.1016/j.jcp.2022.111422 \n";

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

  Point point(r0, v0);
  auto particles = prepare_electron(point);

  dt = 0.16;
  geom_nt = 1000;
  diagnose_period = geom_nt;

  Vector3R check_mean_v;

  PointTrace trace(__FILE__, chin_scheme_id, point, 5);
  BorisPush push;

  if (chin_scheme_id.ends_with("LF"))
    push.update_r(-dt / 2.0, point, *particles);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    PetscCall(trace.diagnose(t));
    process_impl(chin_scheme_id, push, point, *particles, get_magnetic_field);
    check_mean_v += point.p / static_cast<PetscReal>(geom_nt);
  }

  PetscReal alpha = particles->mass(point) / (2.0 * particles->charge(point));
  PetscReal v_util2 = POW2(v0.x()) + 2.0 * POW2(v0.y());
  Vector3R B_p(0, -B_coeff / r0.length(), 0);
  Vector3R B_grad(B_coeff / r0.squared(), 0, 0);
  Vector3R v_gradB = alpha * v_util2 * B_p.cross(B_grad) / POW3(B_p.length());

  PetscCheck(check_mean_v.dot(v_gradB) > 0.0, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Particle should drift along theoretical prediction, having mean: (%f, %f, %f), theory: (%f, %f, %f)", REP3_A(check_mean_v), REP3_A(v_gradB));

  PetscCall(compare_temporal(__FILE__, chin_scheme_id + ".txt"));

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
