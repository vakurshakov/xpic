#include "drift_kinetic_push.h"

#include "src/algorithms/boris_push.h"
#include "src/interfaces/particles.h"

static constexpr char help[] =
  "Test: magnetic mirror with double-Gaussian mirrors.\n"
  "Guiding-center push (analytic/grid) should match the Boris pusher.\n";

using gaussian_magnetic_mirror::get_Bz;
using gaussian_magnetic_mirror::get_fields;
using namespace drift_kinetic_test_utils;

constexpr PetscReal mirror_L = gaussian_magnetic_mirror::L;
constexpr Vector3R mirror_axis{mirror_L, mirror_L, mirror_L};

const auto gaussian_fields = make_translated_field_getter(
  [](const Vector3R& local_pos, Vector3R& B_p, Vector3R& gradB_p) {
    Vector3R E_dummy;
    get_fields(Vector3R{}, local_pos, E_dummy, B_p, gradB_p);
  },
  mirror_axis);

Vector3R get_B_vector(const Vector3R& r)
{
  Vector3R B_p;
  Vector3R gradB_p;
  gaussian_fields(r, B_p, gradB_p);
  return B_p;
}

Vector3R get_gradB_vector(const Vector3R& r)
{
  Vector3R B_p;
  Vector3R gradB_p;
  gaussian_fields(r, B_p, gradB_p);
  return gradB_p;
}

void get_analytical_fields(
  const Vector3R&, const Vector3R& rn, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = {0.0, 0.0, 0.0};
  gaussian_fields(rn, B_p, gradB_p);
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  const PetscReal d = 0.05;
  dt = omega_dt / get_Bz(mirror_L);
  geom_nt = 20'000;
  diagnose_period = geom_nt / 10;
  geom_nx = static_cast<PetscInt>(2 * mirror_L / d);
  geom_ny = static_cast<PetscInt>(2 * mirror_L / d);
  geom_nz = static_cast<PetscInt>(2 * mirror_L / d);
  dx = d;
  dy = d;
  dz = d;

  World::set_geometry(
    geom_nx, geom_ny, geom_nz, geom_nt,
    dx, dy, dz, dt,
    diagnose_period);

  World world;
  PetscCall(world.initialize());

  Vec E_vec;
  Vec B_vec;
  Vec gradB_vec;
  PetscCall(DMCreateGlobalVector(world.da, &E_vec));
  PetscCall(DMCreateGlobalVector(world.da, &B_vec));
  PetscCall(DMCreateGlobalVector(world.da, &gradB_vec));

  PetscCall(initialize_staggered_grid_fields(
    world.da, {dx, dy, dz}, E_vec, B_vec, gradB_vec, get_B_vector, get_gradB_vector));

  Vector3R*** E_arr;
  Vector3R*** B_arr;
  Vector3R*** gradB_arr;
  PetscCall(DMDAVecGetArrayRead(world.da, E_vec, &E_arr));
  PetscCall(DMDAVecGetArrayRead(world.da, B_vec, &B_arr));
  PetscCall(DMDAVecGetArrayRead(world.da, gradB_vec, &gradB_arr));

  auto esirkepov = std::make_unique<DriftKineticEsirkepov>(E_arr, B_arr, nullptr, gradB_arr);

  constexpr PetscReal v_perp = 0.1/ M_SQRT2;
  constexpr PetscReal v_par = 0.1/ M_SQRT2;
  const Vector3R r0(mirror_axis.x() + 0.1, mirror_axis.y(), mirror_axis.z());
  const Vector3R v0(v_perp, 0.0, v_par);

  const PetscReal B_mag = get_B_vector(r0).length();
  const PetscReal rho = v_perp / B_mag;
  Point point_init(r0 - Vector3R{0.0, rho, 0.0}, v0);
  PointByField point_analytical(point_init, get_B_vector(r0), m);
  PointByField point_grid(point_init, get_B_vector(r0), m);

  Point point_boris(r0, v0);

  DriftComparisonStats drift_stats;
  BorisComparisonStats boris_stats;
  boris_stats.mu_reference = point_grid.mu();
  boris_stats.energy_reference = compute_kinetic_energy(point_grid);

  SortParameters sort_parameters{"boris_sort", 1, 1.0, q, m};
  interfaces::Particles particles(world, sort_parameters);

  DriftKineticPush push_analytical;
  push_analytical.set_qm(q / m);
  push_analytical.set_mp(m);
  push_analytical.set_fields_callback(get_analytical_fields);

  DriftKineticPush push_grid;
  push_grid.set_qm(q / m);
  push_grid.set_mp(m);
  push_grid.set_fields_callback([
    &](const Vector3R& r0_local, const Vector3R& rn_local, Vector3R& E_p,
        Vector3R& B_p, Vector3R& gradB_p) {
    esirkepov->interpolate(E_p, B_p, gradB_p, rn_local, r0_local);
  });

  BorisPush push_boris;
  push_boris.set_qm(q / m);

  const auto id = std::format("omega_dt_{:.4f}", omega_dt);
  PointByFieldTrace trace_analytical(__FILE__, id + "_analytical", point_analytical, geom_nt / 1000);
  PointByFieldTrace trace_grid(__FILE__, id + "_grid", point_grid, geom_nt / 1000);
  PointTrace trace_boris(__FILE__, id + "_boris", point_boris, geom_nt / 1000);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_analytical_old = point_analytical;
    const PointByField point_grid_old = point_grid;

    PetscCall(trace_analytical.diagnose(t));
    PetscCall(trace_grid.diagnose(t));
    PetscCall(trace_boris.diagnose(t));

    push_analytical.process(dt, point_analytical, point_analytical_old);
    push_grid.process(dt, point_grid, point_grid_old);
    boris_step(push_boris, point_boris, particles, get_B_vector);

    const PetscReal position_error = (point_analytical.r - point_grid.r).length();
    drift_stats.max_position_error = std::max(drift_stats.max_position_error, position_error);

    Vector3R E_analytical;
    Vector3R B_analytical;
    Vector3R gradB_analytical;
    Vector3R E_grid;
    Vector3R B_grid;
    Vector3R gradB_grid;

    get_analytical_fields(point_analytical_old.r, point_analytical.r,
      E_analytical, B_analytical, gradB_analytical);
    esirkepov->interpolate(E_grid, B_grid, gradB_grid, point_grid.r, point_grid_old.r);

    update_boris_comparison_step(
      point_grid,
      point_boris,
      B_analytical,
      gradB_analytical,
      B_grid,
      gradB_grid,
      get_B_vector,
      drift_stats,
      boris_stats,
      mirror_axis.z());

    if (t % (geom_nt / 10) == 0) {
      PetscCall(log_progress(id.c_str(), t, geom_nt, drift_stats.max_position_error));
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(world.da, E_vec, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(world.da, B_vec, &B_arr));
  PetscCall(DMDAVecRestoreArrayRead(world.da, gradB_vec, &gradB_arr));

  PetscCall(VecDestroy(&E_vec));
  PetscCall(VecDestroy(&B_vec));
  PetscCall(VecDestroy(&gradB_vec));

  PetscCall(particles.finalize());
  drift_stats.simulation_time = dt * geom_nt;
  drift_stats.total_steps = geom_nt;
  drift_stats.final_position_analytical = point_analytical.r;
  drift_stats.final_position_grid = point_grid.r;
  boris_stats.final_position_boris = point_boris.r;

  PetscCall(print_statistics(drift_stats, boris_stats));

  PetscCall(world.finalize());
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
