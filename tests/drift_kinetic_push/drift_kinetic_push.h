#include "src/interfaces/diagnostic.h"
#include "src/interfaces/particles.h"
#include "src/algorithms/boris_push.h"
#include "src/algorithms/crank_nicolson_push.h"
#include "src/algorithms/drift_kinetic_implicit.h"
#include "src/algorithms/drift_kinetic_push.h"
#include "src/utils/utils.h"
#include "src/utils/vector3.h"
#include "src/utils/world.h"
#include "tests/common.h"
#include "src/utils/configuration.h"

constexpr PetscReal q = -1.0;
constexpr PetscReal m = +1.0;

PetscErrorCode get_omega_dt(PetscReal& omega_dt)
{
  PetscFunctionBeginUser;
  PetscBool flg;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-omega_dt", &omega_dt, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must specify the timestep (Omega * dt) with '-omega_dt'");
  PetscFunctionReturn(PETSC_SUCCESS);
}

namespace quadratic_magnetic_mirror {

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal W = 20.0;  // width of the radial well
constexpr PetscReal D = 40.0;  // length of the mirror
constexpr PetscReal Rc = W / 2;
constexpr PetscReal L = D / 2;

PetscReal get_Bz(PetscReal z)
{
  return B_min + (B_max - B_min) * POW2(z / D);
}

PetscReal get_B(PetscReal r, PetscReal z)
{
  return get_Bz(z) * (1.0 + 0.5 * POW2(r / W));
}

PetscReal get_dBz_dz(PetscReal z)
{
  return 2 * (B_max - B_min) * z / POW2(D);
}

void get_fields(const Vector3R&, const Vector3R& pos, //
  Vector3R&, Vector3R& B_p, Vector3R& gradB_p)
{
  PetscReal x = pos.x() - Rc;
  PetscReal y = pos.y() - Rc;
  PetscReal z = pos.z() - L;
  PetscReal r = std::hypot(x, y);

  PetscReal Bz = get_Bz(z);
  PetscReal B = get_B(r, z);

  B_p = Vector3R{0.0, 0.0, B};

  PetscReal dBz_dz = get_dBz_dz(z);
  PetscReal dB_dz = dBz_dz * (1.0 + 0.5 * POW2(r / D));
  PetscReal dB_dr = Bz * r / POW2(D);

  gradB_p = (r > 1e-10) //
    ? Vector3R{x / r * dB_dr, y / r * dB_dr, dB_dz}
    : Vector3R{0.0, 0.0, dB_dz};
}

}  // namespace quadratic_magnetic_mirror

namespace gaussian_magnetic_mirror {

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal L = 5.0;      // Half the length of the trap
constexpr PetscReal W = 1.0;      // Mirror width
constexpr PetscReal S = POW2(W);  // Mirror width squared
constexpr PetscReal Rc = L;

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

  // 3) Calculation of the field modulus gradient |B|
  PetscReal dB_dr = -0.5 * r * d2Bz_dz2;
  PetscReal dB_dz = dBz_dz - 0.25 * r2 * d3Bz_dz3;

  gradB_p = (r > 1e-12) //
    ? Vector3R{x / r * dB_dr, y / r * dB_dr, dB_dz}
    : Vector3R{0, 0, dB_dz};
}

}  // namespace gaussian_magnetic_mirror

namespace drift_kinetic_test_utils {

using Arr = Vector3R***;

void overwrite_config(//
  PetscReal _gx, PetscReal _gy, PetscReal _gz, PetscReal _gt,  //
  PetscReal _dx, PetscReal _dy, PetscReal _dz, PetscReal _dt,  //
  PetscReal _dtp){

  Configuration::overwrite({
    {"Simulation", "eccapfim"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {
      "Geometry",
      {
        {"x", _gx},
        {"y", _gy},
        {"z", _gz},
        {"t", _gt},
        {"dx", _dx},
        {"dy", _dy},
        {"dz", _dz},
        {"dt", _dt},
        {"diagnose_period", _dtp},
        {"da_boundary_x", "DM_BOUNDARY_GHOSTED"},
        {"da_boundary_y", "DM_BOUNDARY_GHOSTED"},
        {"da_boundary_z", "DM_BOUNDARY_GHOSTED"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", 100},
        {"n", +1.0},
        {"q", -1.0},
        {"m", +1.0},
        {"T", +0.1},
      }},
    },
    {
      "Presets",
      {{
        {"command", "SetParticles"},
        {"particles", "electrons"},
        {"coordinate", {{"name", "CoordinateInBox"}}},
        {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
      }},
    },
    {
      "Diagnostics",
      {
        {{"diagnostic", "FieldView"}, {"field", "E"}},
        {{"diagnostic", "FieldView"}, {"field", "B"}},
        {
          {"diagnostic", "DistributionMoment"},
          {"particles", "electrons"},
          {"moment", "density"},
        },
      },
    },
  });
}

class FieldContext {
public:
  World world;
  DM da;

  Vec E_vec = nullptr;
  Vec B_vec = nullptr;
  Vec gradB_vec = nullptr;

  Vec dBdx_vec = nullptr;
  Vec dBdy_vec = nullptr;
  Vec dBdz_vec = nullptr;

  Arr E_arr;
  Arr B_arr;
  Arr gradB_arr;

  Arr dBdx_arr;
  Arr dBdy_arr;
  Arr dBdz_arr;

  template<typename FillFunc>
  PetscErrorCode initialize(FillFunc fill_func)
  {
    PetscFunctionBeginUser;
    PetscCall(world.initialize());
    da = world.da;

    PetscCall(DMCreateLocalVector(da, &E_vec));
    PetscCall(DMCreateLocalVector(da, &B_vec));
    PetscCall(DMCreateLocalVector(da, &gradB_vec));
    PetscCall(DMCreateGlobalVector(da, &dBdx_vec));
    PetscCall(DMCreateGlobalVector(da, &dBdy_vec));
    PetscCall(DMCreateGlobalVector(da, &dBdz_vec));

    PetscCall(DMDAVecGetArrayWrite(world.da, E_vec, &E_arr));
    PetscCall(DMDAVecGetArrayWrite(world.da, B_vec, &B_arr));
    PetscCall(DMDAVecGetArrayWrite(world.da, gradB_vec, &gradB_arr));
    PetscCall(DMDAVecGetArrayWrite(world.da, dBdx_vec, &dBdx_arr));
    PetscCall(DMDAVecGetArrayWrite(world.da, dBdy_vec, &dBdy_arr));
    PetscCall(DMDAVecGetArrayWrite(world.da, dBdz_vec, &dBdz_arr));

    PetscInt xs, ys, zs, xm, ym, zm;
    PetscInt gxs, gys, gzs, gxm, gym, gzm;
    PetscCall(DMDAGetCorners(world.da, &xs, &ys, &zs, &xm, &ym, &zm));
    PetscCall(DMDAGetGhostCorners(world.da, &gxs, &gys, &gzs, &gxm, &gym, &gzm));

    PetscInt i, j, k;

    for (k = world.start[Z]; k < world.end[Z]+1; ++k) {
      for (j = world.start[Y]; j < world.end[Y]+1; ++j) {
        for (i = world.start[X]; i < world.end[X]+1; ++i) {
          fill_func(i, j, k, E_arr[k][j][i], B_arr[k][j][i], gradB_arr[k][j][i]);
        }
      }
    }

    PetscCall(DMDAVecRestoreArrayWrite(world.da, E_vec, &E_arr));
    PetscCall(DMDAVecRestoreArrayWrite(world.da, B_vec, &B_arr));
    PetscCall(DMDAVecRestoreArrayWrite(world.da, gradB_vec, &gradB_arr));

    PetscCall(DMDAVecGetArrayRead(world.da, E_vec, &E_arr));
    PetscCall(DMDAVecGetArrayRead(world.da, B_vec, &B_arr));
    PetscCall(DMDAVecGetArrayRead(world.da, gradB_vec, &gradB_arr));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode finalize()
  {
    PetscFunctionBeginUser;
    PetscCall(DMDAVecRestoreArrayRead(world.da, E_vec, &E_arr));
    PetscCall(DMDAVecRestoreArrayRead(world.da, B_vec, &B_arr));
    PetscCall(DMDAVecRestoreArrayRead(world.da, gradB_vec, &gradB_arr));
    PetscCall(DMDAVecRestoreArrayWrite(world.da, dBdx_vec, &dBdx_arr));
    PetscCall(DMDAVecRestoreArrayWrite(world.da, dBdy_vec, &dBdy_arr));
    PetscCall(DMDAVecRestoreArrayWrite(world.da, dBdz_vec, &dBdz_arr));

    PetscCall(VecDestroy(&E_vec));
    PetscCall(VecDestroy(&B_vec));
    PetscCall(VecDestroy(&gradB_vec));
    PetscCall(VecDestroy(&dBdx_vec));
    PetscCall(VecDestroy(&dBdy_vec));
    PetscCall(VecDestroy(&dBdz_vec));

    PetscCall(world.finalize());
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

class TraceTriplet : public interfaces::Diagnostic {
public:
  PointByFieldTrace trace_analytical;
  PointByFieldTrace trace_grid;
  PointTrace trace_boris;

  TraceTriplet(std::string_view file, std::string id, PetscInt skip,
    const PointByField& point_analytical, //
    const PointByField& point_grid, //
    const Point& point_boris)
    : trace_analytical(file, id + "_analytical", point_analytical, skip),
      trace_grid(file, id + "_grid", point_grid, skip),
      trace_boris(file, id + "_boris", point_boris, skip)
  {
  }

  PetscErrorCode diagnose(PetscInt t) override
  {
    PetscFunctionBeginUser;
    PetscCall(trace_analytical.diagnose(t));
    PetscCall(trace_grid.diagnose(t));
    PetscCall(trace_boris.diagnose(t));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

struct ComparisonStats {
  PetscReal ref_mu = 0.0;
  PetscReal ref_energy = 0.0;

  PetscReal max_err_B = 0.0;
  PetscReal max_err_gradB = 0.0;
  PetscReal max_err_pos = 0.0;
  PetscReal max_err_z = 0.0;
  PetscReal max_err_par = 0.0;
  PetscReal max_err_mu = 0.0;
  PetscReal max_err_energy = 0.0;

  Vector3R final_pos_analytical;
  Vector3R final_pos_grid;
  Vector3R final_pos_boris;
};

PetscReal get_kinetic_energy(const PointByField& point)
{
  return 0.5 * (POW2(point.p_perp) + POW2(point.p_parallel));
}

PetscReal get_kinetic_energy(const Point& point)
{
  return 0.5 * point.p.squared();
}

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

void update_comparison_stats(ComparisonStats& stats,
  const PointByField& point_analytical, //
  const PointByField& point_grid,       //
  const Point& point_boris,             //
  const Vector3R& B_analytical,         //
  const Vector3R& gradB_analytical,     //
  const Vector3R& B_grid,               //
  const Vector3R& gradB_grid)
{
  PetscReal err_B = (B_analytical - B_grid).length();
  stats.max_err_B = std::max(stats.max_err_B, err_B);

  PetscReal err_gradB = (gradB_analytical - gradB_grid).length();
  stats.max_err_gradB = std::max(stats.max_err_gradB, err_gradB);

  PetscReal err_pos = (point_analytical.r - point_grid.r).length();
  stats.max_err_pos = std::max(stats.max_err_pos, err_pos);

  PetscReal err_z = std::abs(point_grid.z() - point_boris.z());
  stats.max_err_z = std::max(stats.max_err_z, err_z);

  Vector3R B = B_analytical;

  PetscReal v_par = point_boris.p.parallel_to(B).length();
  PetscReal err_parallel = std::abs(point_grid.p_par() - v_par);
  stats.max_err_par = std::max(stats.max_err_par, err_parallel);

  PetscReal p_perp = point_boris.p.transverse_to(B).length();
  PetscReal mu = 0.5 * m * POW2(p_perp) / B.length();
  PetscReal err_mu = std::abs(point_grid.mu() - mu);
  stats.max_err_mu = std::max(stats.max_err_mu, err_mu);

  PetscReal energy_drift = get_kinetic_energy(point_grid);
  PetscReal energy_boris = get_kinetic_energy(point_boris);
  PetscReal err_energy = std::abs(energy_drift - energy_boris);
  stats.max_err_energy = std::max(stats.max_err_energy, err_energy);
}

PetscErrorCode print_statistics(ComparisonStats& stats,
  const PointByField& point_analytical, //
  const PointByField& point_grid, //
  const Point& point_boris)
{
  PetscFunctionBeginUser;
  stats.final_pos_analytical = point_analytical.r;
  stats.final_pos_grid = point_grid.r;
  stats.final_pos_boris = point_boris.r;

#if 1
  LOG("Simulation time: {:.6e}", geom_t);
  LOG("Total steps: {}", geom_nt);

  LOG("\n=== DRIFT VS GRID STATISTICS ===");
  LOG("Field errors comparison:");
  LOG("  Max B field error:     {:.8e}", stats.max_err_B);
  LOG("  Max gradB field error: {:.8e}", stats.max_err_gradB);
  LOG("Trajectory comparison:");
  LOG("  Max position error:    {:.8e}", stats.max_err_pos);
  LOG("  Final pos analytical:  ({:.6e} {:.6e} {:.6e})", REP3_A(stats.final_pos_analytical));
  LOG("  Final pos grid:        ({:.6e} {:.6e} {:.6e})", REP3_A(stats.final_pos_grid));

  LOG("\n=== DRIFT VS BORIS STATISTICS ===");
  LOG("Reference mu:      {:.8e}", stats.ref_mu);
  LOG("Reference energy:  {:.8e}", stats.ref_energy);
  LOG("Max z error:       {:.8e}", stats.max_err_z);
  LOG("Max p_parallel err {:.8e}", stats.max_err_par);
  LOG("Max mu error:      {:.8e}", stats.max_err_mu);
  LOG("Max energy error:  {:.8e}", stats.max_err_energy);
  LOG("Final pos boris:   ({:.6e} {:.6e} {:.6e})", REP3_A(stats.final_pos_boris));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace drift_kinetic_test_utils
