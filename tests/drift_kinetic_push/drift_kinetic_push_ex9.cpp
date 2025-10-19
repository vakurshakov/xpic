#include "drift_kinetic_push.h"

#include "src/algorithms/boris_push.h"
#include "src/algorithms/crank_nicolson_push.h"

static constexpr char help[] =
  "Magnetic mirror field with realistic analytic mirrors, includes the\n"
  "comparison between different pushers. Sweep in critical pitch angle\n"
  "fraction and pusher time step (in Omega * dt units) is added.\n";

PetscReal pitch_frac = 1.005;
PetscReal Omega_dt = 0.1;

auto format(const char* push)
{
  return std::format("{}_omega_dt_{:.4f}_pf_{:.3f}", push, Omega_dt, pitch_frac);
}

int main(int argc, char** argv)
{
  using namespace gaussian_magnetic_mirror;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-omega_dt", &Omega_dt, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-pitch_frac", &pitch_frac, NULL));

  PetscReal mirror_R = get_Bz(0) / get_Bz(L);

  PetscReal v_abs = 0.1;
  PetscReal v_pitch = pitch_frac * asin(sqrt(mirror_R));
  PetscReal v_par = v_abs * cos(v_pitch);
  PetscReal v_perp = v_abs * sin(v_pitch);

  Vector3R r0(0.1, 0, 0);
  Vector3R v0(v_perp, 0, v_par);

  PetscReal Omega = get_Bz(r0.z());
  dt = Omega_dt / Omega;
  geom_nt = 5000;
  diagnose_period = 1;

  Vector3R B{
    -0.5 * r0.x() * get_dBz_dz(r0.z()),
    -0.5 * r0.y() * get_dBz_dz(r0.z()),
    get_Bz_corr(r0),
  };

#if 0
  Vector3R gradB{
    -0.5 * r0.x() * get_d2Bz_dz2(r0.z()),
    -0.5 * r0.y() * get_d2Bz_dz2(r0.z()),
    get_dBz_dz(r0.z()) - 0.25 * hypot(r0.x(), r0.y()) * get_d3Bz_dz3(r0.z()),
  };

  PetscReal rho = v_perp / B.length();

  LOG("Mirror parameters:");
  LOG("  half-length = {}, width = {}", L, D);
  LOG("  field = {}, {}, {}", B.x(), B.y(), B.z());
  LOG("  field gradients = {}, {}, {}", abs(gradB.x() / B.x()), abs(gradB.y() / B.y()), abs(gradB.z() / B.z()));
  LOG("Particle parameters:");
  LOG("  larmor radius = {}", rho);
  LOG("  drift approximation = {}, {}, {}", rho * abs(gradB.x() / B.x()), rho * abs(gradB.y() / B.y()), rho * abs(gradB.z() / B.z()));
#endif

  Vector3R dB_p;

  Point b_p(r0, v0);
  BorisPush b_push;
  b_push.set_qm(1.0);

  Point cn_p(r0, v0);
  CrankNicolsonPush cn_push;
  cn_push.set_qm(1.0);
  cn_push.set_fields_callback(
    [&](const Vector3R& r0, const Vector3R& r1, Vector3R& E, Vector3R& B) {
      get_fields(r0, (r1 + r0) / 2, E, B, dB_p);
    });

  PointByField dk_p({r0, v0}, {0.0, 0.0, get_Bz_corr(r0)}, 1.0, q/m);
  DriftKineticPush dk_push;
  dk_push.set_qm(1.0);
  dk_push.set_mp(1.0);
  dk_push.set_fields_callback(get_fields);

  PointTrace b_d(__FILE__, format("boris"), b_p);
  PointTrace cn_d(__FILE__, format("crank_nicolson"), cn_p);
  PointByFieldTrace dk_d(__FILE__, format("drift_kinetic"), dk_p);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    Vector3R E_p, B_p;
    b_push.update_r(dt / 2.0, b_p);
    get_fields(b_p.r, b_p.r, E_p, B_p, dB_p);
    b_push.set_fields(E_p, B_p);
    b_push.update_vB(dt, b_p);
    b_push.update_r(dt / 2.0, b_p);
    PetscCall(b_d.diagnose(t));

    Point cn_p0 = cn_p;
    cn_push.process(dt, cn_p, cn_p0);
    PetscCall(cn_d.diagnose(t));

    PointByField dk_p0 = dk_p;
    dk_push.process(dt, dk_p, dk_p0);
    PetscCall(dk_d.diagnose(t));
  }

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
