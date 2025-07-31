#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: Magnetic mirror field with realistic analytic mirrors.\n"
  "Particle should remain trapped in the well; energy conserved.\n";

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal L = 5.0;  // Half the length of the trap
constexpr PetscReal D = 1.0;  // Mirror width squared

inline PetscReal exp(PetscReal z, PetscReal z0)
{
  return std::exp(-POW2(z - z0) / D);
}

// Center field profile on the axis (double gauss)
PetscReal get_Bz(PetscReal z)
{
  return B_min + (B_max - B_min) * (exp(z, -L) + exp(z, +L));
}

PetscReal get_dBz_dz(PetscReal z)
{
  return (B_max - B_min) * //
    ((-2.0 * (z + L) / D * exp(z, -L)) + //
      (-2.0 * (z - L) / D * exp(z, +L)));
}

PetscReal get_d2Bz_dz2(PetscReal z)
{
  PetscReal t1 = (z + L);
  PetscReal t2 = (z - L);
  return (B_max - B_min) * //
    ((-2.0 / D + 4.0 * POW2(t1 / D)) * exp(z, -L) + //
      (-2.0 / D + 4.0 * POW2(t2 / D)) * exp(z, +L));
}

PetscReal get_d3Bz_dz3(PetscReal z)
{
  PetscReal t1 = (z + L);
  PetscReal t2 = (z - L);
  return (B_max - B_min) * //
    ((12.0 * t1 / (D * D) - 8.0 * POW3(t1 / D)) * exp(z, -L) + //
      (12.0 * t2 / (D * D) - 8.0 * POW3(t2 / D)) * exp(z, +L));
}

// Bz field off axis
PetscReal get_Bz_corr(const Vector3R& r)
{
  return get_Bz(r.z()) - 0.25 * (POW2(r.x()) + POW2(r.y())) * get_d2Bz_dz2(r.z());
}

void get_fields(const Vector3R& pos, Vector3R&, Vector3R& B_p, Vector3R& gradB_p)
{
  PetscReal x = pos.x();
  PetscReal y = pos.y();
  PetscReal z = pos.z();
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

  // 3) Calculation of the field modulus gradient |B|
  PetscReal dB_dr = -0.5 * r * d2Bz_dz2;
  PetscReal dB_dz = dBz_dz - 0.25 * r2 * d3Bz_dz3;

  gradB_p = (r > 1e-12) //
    ? Vector3R{(x / r) * dB_dr, (y / r) * dB_dr, dB_dz}
    : Vector3R{0, 0, dB_dz};
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  constexpr PetscReal v_perp = 1.0;
  constexpr PetscReal v_par = 0.6;
  constexpr Vector3R r0(0.5, 0.0, 0.0);
  constexpr Vector3R v0(v_perp, 0.0, v_par);
  Point point_init(r0, v0);
  PointByField point_n(point_init, {0.0, 0.0, get_Bz_corr(r0)}, 1.0);

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dt = omega_dt / get_Bz(r0.z());
  geom_nt = 300'000;
  diagnose_period = geom_nt / 2;

  auto id = std::format("omega_dt_{:.1f}", omega_dt);
  PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 123);

  DriftKineticPush push;
  push.set_qm(1.0);
  push.set_mp(1.0);
  push.set_fields_callback(get_fields);

  PetscReal z_max = L + 1e-2;
  PetscReal r_max = L + 1e-2;

  const PetscReal old_E = //
    point_n.p_parallel * point_n.p_parallel + //
    point_n.p_perp * point_n.p_perp;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_0 = point_n;
    push.process(dt, point_n, point_0);

    PetscCall(trace.diagnose(t));

    PetscCheck(std::abs(point_n.r.z()) <= z_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle escaped mirror! z = %.6e, allowed = %.6e", point_n.r.z(), z_max);

    PetscCheck(point_n.r.length() <= r_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle escaped radial well! r = %.6e, allowed = %.6e", point_n.r.length(), r_max);
  }

  const PetscReal new_E = //
    point_n.p_parallel * point_n.p_parallel + //
    point_n.p_perp * point_n.p_perp;

  PetscCheck(equal_tol(new_E, old_E, 1e-6), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Energy not conserved: new = %.6e, old = %.6e", new_E, old_E);

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
