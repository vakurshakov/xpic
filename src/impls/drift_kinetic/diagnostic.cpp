#include "diagnostic.h"

#include "src/impls/drift_kinetic/simulation.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/utils/geometries.h"
#include "src/utils/operators.h"
#include "src/utils/shape.h"
#include "src/utils/vector_utils.h"

namespace drift_kinetic {

namespace {

struct ChargeShape {
  static constexpr PetscReal shr = 1.5;
  static constexpr PetscInt shw = (PetscInt)(2.0 * shr);
  static constexpr PetscInt shm = POW3(shw);
  static constexpr PetscReal (&sfunc)(PetscReal) = spline_of_2nd_order;

  Vector3I start;
  PetscReal cache[shm];

  void setup(const Vector3R& r)
  {
    Vector3R p_r = ::Shape::make_r(r);

    start = Vector3I{
      (PetscInt)(std::ceil(p_r[X] - shr)),
      (PetscInt)(std::ceil(p_r[Y] - shr)),
      (PetscInt)(std::ceil(p_r[Z] - shr)),
    };

#pragma omp simd
    for (PetscInt i = 0; i < shm; ++i) {
      const auto g_x = (PetscReal)(start[X] + i % shw);
      const auto g_y = (PetscReal)(start[Y] + (i / shw) % shw);
      const auto g_z = (PetscReal)(start[Z] + (i / shw) / shw);
      cache[i] = sfunc(p_r[X] - g_x) * sfunc(p_r[Y] - g_y) * sfunc(p_r[Z] - g_z);
    }
  }
};

}  // namespace

namespace {

// In the drift-kinetic representation, a per-particle "p" stored in
// `PointByField` carries the gyrocenter velocity components (after
// `form_iteration` writes Vph = (r_n - r_0)/dt into `point.p`); `p_parallel`
// and `p_perp` are velocity magnitudes along/across B. With the convention
// `p == v` used in the kinetic-energy formula in particles.cpp, the raw
// second-moment quantities below have units of mass*velocity^2, i.e. twice
// the per-direction kinetic energy density per particle.

inline std::vector<PetscReal> dk_get_velocity(
  const Particles&, const PointByField& point, PetscReal)
{
  return {point.p[X], point.p[Y], point.p[Z]};
}

inline std::vector<PetscReal> dk_get_velocity_parallel(
  const Particles&, const PointByField& point, PetscReal)
{
  return {point.p_parallel};
}

inline std::vector<PetscReal> dk_get_temperature_parallel(
  const Particles& particles, const PointByField& point, PetscReal)
{
  return {particles.parameters.m * POW2(point.p_parallel)};
}

inline std::vector<PetscReal> dk_get_temperature_perp(
  const Particles&, const PointByField& point, PetscReal lenB)
{
  // Perpendicular temperature from the adiabatic invariant: 2 * mu * B =
  // m * v_perp^2, evaluated with the current B at the gyrocenter (mu is
  // conserved while p_perp goes stale as B evolves after the push).
  return {2.0 * point.mu_p * lenB};
}

// Project the gyrocenter velocity Vph (stored in `point.p` after
// `form_iteration`) onto cylindrical components at the gyrocenter position.
// Follows the same convention as `_get_v_cyl` in distribution_moment.cpp:
// particles inside the singular cell (r=0) keep Cartesian components.
inline Vector3R dk_v_cyl(const PointByField& point)
{
  PetscReal x = point.x() - 0.5 * geom_x;
  PetscReal y = point.y() - 0.5 * geom_y;
  PetscReal r = std::hypot(x, y);

  const auto& v = point.p;
  if (std::isinf(1.0 / r))
    return v;

  return {
    (+x * v[X] + y * v[Y]) / r,
    (-y * v[X] + x * v[Y]) / r,
    v[Z],
  };
}

inline std::vector<PetscReal> dk_get_momentum_flux(
  const Particles& particles, const PointByField& point, PetscReal)
{
  const PetscReal m = particles.parameters.m;
  const auto& v = point.p;
  return {
    m * v[X] * v[X],
    m * v[X] * v[Y],
    m * v[X] * v[Z],
    m * v[Y] * v[Y],
    m * v[Y] * v[Z],
    m * v[Z] * v[Z],
  };
}

inline std::vector<PetscReal> dk_get_momentum_flux_diag(
  const Particles& particles, const PointByField& point, PetscReal)
{
  const PetscReal m = particles.parameters.m;
  const auto& v = point.p;
  return {
    m * v[X] * v[X],
    m * v[Y] * v[Y],
    m * v[Z] * v[Z],
  };
}

inline std::vector<PetscReal> dk_get_momentum_flux_cyl(
  const Particles& particles, const PointByField& point, PetscReal)
{
  const PetscReal m = particles.parameters.m;
  const Vector3R v = dk_v_cyl(point);
  return {
    m * v[R] * v[R],
    m * v[R] * v[A],
    m * v[R] * v[Z],
    m * v[A] * v[A],
    m * v[A] * v[Z],
    m * v[Z] * v[Z],
  };
}

inline std::vector<PetscReal> dk_get_momentum_flux_diag_cyl(
  const Particles& particles, const PointByField& point, PetscReal)
{
  const PetscReal m = particles.parameters.m;
  const Vector3R v = dk_v_cyl(point);
  return {
    m * v[R] * v[R],
    m * v[A] * v[A],
    m * v[Z] * v[Z],
  };
}

}  // namespace

DkMoment dk_moment_from_string(const std::string& name)
{
  if (name == "velocity")
    return dk_get_velocity;
  if (name == "velocity_parallel")
    return dk_get_velocity_parallel;
  if (name == "temperature_parallel")
    return dk_get_temperature_parallel;
  if (name == "temperature_perp")
    return dk_get_temperature_perp;
  if (name == "momentum_flux")
    return dk_get_momentum_flux;
  if (name == "momentum_flux_diag")
    return dk_get_momentum_flux_diag;
  if (name == "momentum_flux_cyl")
    return dk_get_momentum_flux_cyl;
  if (name == "momentum_flux_diag_cyl")
    return dk_get_momentum_flux_diag_cyl;
  return nullptr;
}

struct DkShape {
  static constexpr PetscInt shr = 1;
  static constexpr PetscInt shw = 2;
  static constexpr PetscInt shm = POW3(shw);
  static constexpr const auto& sfunc = spline_of_1st_order;

  Vector3I start;
  PetscReal cache[shm];

  void setup(const Vector3R& r)
  {
    Vector3R p_r = ::Shape::make_r(r);
    start = ::Shape::make_start(p_r, shr);

#pragma omp simd
    for (PetscInt i = 0; i < shm; ++i) {
      auto g_x = (PetscReal)(start[X] + i % shw) + 0.5;
      auto g_y = (PetscReal)(start[Y] + (i / shw) % shw) + 0.5;
      auto g_z = (PetscReal)(start[Z] + (i / shw) / shw) + 0.5;
      cache[i] = sfunc(p_r[X] - g_x) * sfunc(p_r[Y] - g_y) * sfunc(p_r[Z] - g_z);
    }
  }
};

std::unique_ptr<DkDistributionMoment> DkDistributionMoment::create(
  const std::string& out_dir, const Particles& particles,
  const Moment& moment, const Region& region)
{
  PetscFunctionBeginUser;
  MPI_Comm newcomm;
  PetscCallAbort(PETSC_COMM_WORLD, World::create_local_comm(particles.world.da, region, &newcomm));
  if (newcomm == MPI_COMM_NULL)
    PetscFunctionReturn(nullptr);

  // The shared DistributionMomentBuilder hands us a CWD-relative path; prefix
  // it with the simulation's output directory so density frames land next to
  // the field diagnostics regardless of where the binary was launched from.
  std::string full_out_dir = CONFIG().out_dir + "/" + out_dir;

  auto* diagnostic = new DkDistributionMoment(full_out_dir, particles, moment, newcomm);
  PetscCallAbort(PETSC_COMM_WORLD, diagnostic->set_data_views(region));
  PetscFunctionReturn(std::unique_ptr<DkDistributionMoment>(diagnostic));
}

std::unique_ptr<DkDistributionMoment> DkDistributionMoment::create(
  const std::string& out_dir, const Particles& particles,
  const DkMoment& moment, const Region& region)
{
  PetscFunctionBeginUser;
  MPI_Comm newcomm;
  PetscCallAbort(PETSC_COMM_WORLD, World::create_local_comm(particles.world.da, region, &newcomm));
  if (newcomm == MPI_COMM_NULL)
    PetscFunctionReturn(nullptr);

  std::string full_out_dir = CONFIG().out_dir + "/" + out_dir;

  auto* diagnostic = new DkDistributionMoment(full_out_dir, particles, moment, newcomm);
  PetscCallAbort(PETSC_COMM_WORLD, diagnostic->set_data_views(region));
  PetscFunctionReturn(std::unique_ptr<DkDistributionMoment>(diagnostic));
}

DkDistributionMoment::DkDistributionMoment(const std::string& out_dir,
  const Particles& particles, const Moment& moment, MPI_Comm newcomm)
  : ::DistributionMoment(out_dir, particles, moment, newcomm),
    dk_particles(particles)
{
}

DkDistributionMoment::DkDistributionMoment(const std::string& out_dir,
  const Particles& particles, const DkMoment& moment, MPI_Comm newcomm)
  : ::DistributionMoment(out_dir, particles, nullptr, newcomm),
    dk_particles(particles),
    dk_moment(moment)
{
}

PetscErrorCode DkDistributionMoment::collect()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(field_loc, 0.0));
  PetscCall(VecSet(field, 0.0));

  PetscReal**** arr;
  PetscCall(DMDAVecGetArrayDOFWrite(da, field_loc, &arr));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da_glob, REP3_A(&start), REP3_A(&size)));

  const Vector3I gstart = vector_cast(region.start);
  const Vector3I gsize = vector_cast(region.size);

  DkShape shape;

  // Итерация по дрейфово-кинетическому хранилищу dk_curr_storage
  const auto& dk_storage = dk_particles.get_dk_curr_storage();

  // DK moments receive |B| interpolated at the gyrocenter; check out the
  // current B field once before the particle loop. (The non-DK `moment`
  // path never touches B.)
  Arr B_arr_local = nullptr;
  const auto& sim = dk_particles.simulation();
  if (dk_moment) {
    PetscCall(DMGlobalToLocal(sim.da, sim.B, INSERT_VALUES, sim.B_loc));
    PetscCall(DMDAVecGetArrayRead(sim.da, sim.B_loc,
      reinterpret_cast<void***>(&B_arr_local)));
  }

#pragma omp parallel for private(shape)
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    Vector3I vg{
      start[X] + g % size[X],
      start[Y] + (g / size[X]) % size[Y],
      start[Z] + (g / size[X]) / size[Y],
    };

    if (!is_point_within_bounds(vg, gstart, gsize))
      continue;

    for (const auto& point : dk_storage[g]) {
      shape.setup(point.r);

      ::Point dummy_point(point.r, point.p);

      std::vector<PetscReal> moments;
      if (dk_moment) {
        // Interpolate |B| at the gyrocenter with the 2nd-order spline (same
        // kernel as DriftKineticEsirkepov::interpolate_B).
        Vector3R B_p{};
        ::Shape sh_B;
        sh_B.setup(point.r, 1.5, spline_of_2nd_order);
        SimpleInterpolation interp(sh_B);
        SimpleInterpolation::Context B_ctx{{B_p, B_arr_local}};
        interp.process({}, B_ctx);
        moments = dk_moment(dk_particles, point, B_p.length());
      } else {
        moments = moment(particles, dummy_point);
      }
      auto msize = static_cast<PetscInt>(moments.size());

      for (PetscInt i = 0; i < shape.shm; ++i) {
        PetscInt g_x = shape.start[X] + i % shape.shw;
        PetscInt g_y = shape.start[Y] + (i / shape.shw) % shape.shw;
        PetscInt g_z = shape.start[Z] + (i / shape.shw) / shape.shw;

        PetscReal si = shape.cache[i] * particles.n_Np(dummy_point);

        for (PetscInt j = 0; j < msize; ++j) {
          PetscReal mj = moments[j] * si;

#pragma omp atomic update
          arr[g_z][g_y][g_x][j] += mj;
        }
      }
    }
  }

  if (dk_moment) {
    PetscCall(DMDAVecRestoreArrayRead(sim.da, sim.B_loc,
      reinterpret_cast<void***>(&B_arr_local)));
  }

  PetscCall(DMDAVecRestoreArrayDOFWrite(da, field_loc, &arr));
  PetscCall(DMLocalToGlobal(da, field_loc, ADD_VALUES, field));
  PetscFunctionReturn(PETSC_SUCCESS);
}

std::unique_ptr<MatMultFieldView> MatMultFieldView::create(
  const std::string& out_dir, DM da, Vec source, Mat op,
  const Region& region)
{
  PetscFunctionBeginUser;
  MPI_Comm newcomm;
  PetscCallAbort(PETSC_COMM_WORLD,
    World::create_local_comm(da, region, &newcomm));
  if (newcomm == MPI_COMM_NULL)
    PetscFunctionReturn(nullptr);

  // Owned output Vec lives on the same DM as `source`; `op` acts on it
  // through MatMult during each diagnose() call.
  Vec field = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &field));

  auto* diagnostic = new MatMultFieldView(
    out_dir, da, field, source, op, newcomm);
  PetscCallAbort(PETSC_COMM_WORLD, diagnostic->set_data_views(region));
  PetscFunctionReturn(std::unique_ptr<MatMultFieldView>(diagnostic));
}

MatMultFieldView::MatMultFieldView(const std::string& out_dir, DM da,
  Vec field, Vec source, Mat op, MPI_Comm newcomm)
  : ::FieldView(out_dir, da, field, newcomm),
    source(source),
    op(op)
{
}

PetscErrorCode MatMultFieldView::finalize()
{
  PetscFunctionBeginUser;
  if (field != nullptr)
    PetscCall(VecDestroy(&field));
  PetscCall(::FieldView::finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultFieldView::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(MatMult(op, source, field));
  PetscCall(::FieldView::diagnose(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PointByFieldTrace::PointByFieldTrace(const std::string& out_dir, const Particles& particles, PetscInt skip)
  : TableDiagnostic(out_dir + "/temporal/particle_trace.txt"), skip(skip), particles(particles)
{
}

PetscErrorCode PointByFieldTrace::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(TableDiagnostic::initialize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PointByFieldTrace::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(TableDiagnostic::finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PointByFieldTrace::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t % skip != 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TableDiagnostic::diagnose(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PointByFieldTrace::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;

  const auto& storage = particles.get_dk_curr_storage();
  bool found = false;
  PointByField point;

  for (const auto& cell_list : storage) {
    if (!cell_list.empty()) {
      point = cell_list.front();
      found = true;
      break;
    }
  }

  if (found) {
    add(24, "t_[1/wpe]", "{: .15e}", t * dt);
    add(24, "x_[c/wpe]", "{: .15e}", point.x());
    add(24, "y_[c/wpe]", "{: .15e}", point.y());
    add(24, "z_[c/wpe]", "{: .15e}", point.z());
    add(24, "p_par_[mc]", "{: .15e}", point.p_par());
    add(24, "p_perp_[mc]", "{: .15e}", point.p_perp_ref());
    add(24, "mu_p_[mc^2/B]", "{: .15e}", point.mu());
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

EnergyConservation::EnergyConservation(const Simulation& simulation)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/dk_diagnostic.txt"),
    simulation(simulation)
{
}

void EnergyConservation::calculate_kinetic_energies(
  std::vector<PetscReal>& per_sort, PetscReal& total) const
{
  per_sort.assign(simulation.particles_.size(), 0.0);
  total = 0.0;

  for (PetscInt i = 0; i < (PetscInt)simulation.particles_.size(); ++i) {
    const PetscReal kinetic = simulation.particles_[i]->kinetic_energy_local();
    per_sort[i] = kinetic;
    total += kinetic;
  }
}

PetscErrorCode EnergyConservation::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (charge_da == nullptr && t != 0) {
    PetscCall(init_charge_conservation());
    PetscCall(collect_charge_densities());
  }


  if (!initialized) {
    PetscCall(VecNorm(simulation.E, NORM_2, &w_E));
    PetscCall(VecNorm(simulation.B, NORM_2, &w_B));
    PetscCall(VecDot(simulation.M, simulation.B, &a_MB));
    w_E = 0.5 * POW2(w_E);
    w_B = 0.5 * POW2(w_B);
    calculate_kinetic_energies(K_by_sort, K);
    initialized = true;
  }

  w_E0 = w_E;
  w_B0 = w_B;
  K0 = K;
  K0_by_sort = K_by_sort;
  PetscCall(TableDiagnostic::diagnose(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(init_charge_conservation());
  PetscCall(collect_charge_densities());
  PetscCall(TableDiagnostic::initialize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::finalize()
{
  PetscFunctionBeginUser;
  for (auto& vec : charge_locals)
    PetscCall(VecDestroy(&vec));
  for (auto& vec : charge_fields)
    PetscCall(VecDestroy(&vec));

  charge_locals.clear();
  charge_fields.clear();
  current_densities.clear();

  PetscCall(MatDestroy(&divE));
  PetscCall(DMDestroy(&charge_da));
  PetscCall(TableDiagnostic::finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::init_charge_conservation()
{
  PetscFunctionBeginUser;
  if (charge_da != nullptr)
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscInt g_size[3];
  PetscInt procs[3];
  PetscInt s;
  DMBoundaryType bounds[3];
  DMDAStencilType st;
  PetscCall(DMDAGetInfo(simulation.da, nullptr, REP3_A(&g_size), REP3_A(&procs), nullptr,
    &s, REP3_A(&bounds), &st));

  const PetscInt* ownership[3];
  PetscCall(DMDAGetOwnershipRanges(simulation.da, REP3_A(&ownership)));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(bounds), st, REP3_A(g_size),
    REP3_A(procs), 1, s, REP3_A(ownership), &charge_da));
  PetscCall(DMSetUp(charge_da));

  current_densities.reserve(simulation.particles_.size() + 1);
  charge_locals.reserve(simulation.particles_.size());
  charge_fields.reserve(simulation.particles_.size());

  for (const auto& sort : simulation.particles_) {
    Vec local = nullptr;
    Vec field = nullptr;

    PetscCall(DMCreateLocalVector(charge_da, &local));
    PetscCall(DMCreateGlobalVector(charge_da, &field));

    charge_locals.emplace_back(local);
    charge_fields.emplace_back(field);
    current_densities.emplace_back(sort->J);
  }
  current_densities.emplace_back(simulation.J);

  PetscCheckAbort(current_densities.size() == charge_fields.size() + 1, PETSC_COMM_WORLD,
    PETSC_ERR_USER,
    "Number of `current_densities` should be one more than charge fields");

  Divergence divergence(simulation.da);
  PetscCall(divergence.create_negative(&divE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::collect_charge_density(PetscInt sort_id)
{
  PetscFunctionBeginUser;
  PetscCheck(sort_id >= 0 &&
      sort_id < (PetscInt)simulation.particles_.size(), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Invalid drift-kinetic sort index %" PetscInt_FMT, sort_id);

  auto& local = charge_locals[sort_id];
  auto& field = charge_fields[sort_id];
  const auto& sort = *simulation.particles_[sort_id];
  const interfaces::Particles& particles = sort;
  const auto& dk_curr_storage = sort.get_dk_curr_storage();

  PetscCall(VecSet(local, 0.0));
  PetscCall(VecSet(field, 0.0));

  PetscReal*** arr = nullptr;
  PetscCall(DMDAVecGetArrayWrite(charge_da, local, &arr));

  ChargeShape shape;

#pragma omp parallel for private(shape)
  for (PetscInt g = 0; g < (PetscInt)dk_curr_storage.size(); ++g) {
    for (const auto& point : dk_curr_storage[g]) {
      shape.setup(point.r);

      ::Point equivalent_point(point.r, Vector3R{});
      const PetscReal qn_np = particles.qn_Np(equivalent_point);

      for (PetscInt i = 0; i < shape.shm; ++i) {
        const PetscInt g_x = shape.start[X] + i % shape.shw;
        const PetscInt g_y = shape.start[Y] + (i / shape.shw) % shape.shw;
        const PetscInt g_z = shape.start[Z] + (i / shape.shw) / shape.shw;

#pragma omp atomic update
        arr[g_z][g_y][g_x] += qn_np * shape.cache[i];
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(charge_da, local, &arr));
  PetscCall(DMLocalToGlobal(charge_da, local, ADD_VALUES, field));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::collect_charge_densities()
{
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < (PetscInt)simulation.particles_.size(); ++i)
    PetscCall(collect_charge_density(i));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  add(6, "Time", "{:d}", t);
  for (const auto& sort : simulation.particles_) {
    const auto& name = sort->parameters.sort_name;
    add(16, "MaxPushIt_" + name, "{:d}", sort->get_max_iteration_number());
  }
  add(16, "AvgFieldIt", "{:d}", simulation.last_field_itnum);

  PetscCall(VecNorm(simulation.E, NORM_2, &w_E));
  PetscCall(VecNorm(simulation.B, NORM_2, &w_B));
  PetscCall(VecNorm(simulation.M, NORM_2, &w_M));
  PetscCall(VecDot(simulation.E_hk, simulation.J, &a_EJ));
  PetscCall(VecDot(simulation.M, simulation.B, &a_MB));
  w_E = 0.5 * POW2(w_E);
  w_B = 0.5 * POW2(w_B);
  calculate_kinetic_energies(K_by_sort, K);

  dF = (w_E - w_E0) + (w_B - w_B0);

  add(24, "dK", "{: .16e}", (K - K0));
  for (PetscInt i = 0; i < (PetscInt)K_by_sort.size(); ++i) {
    const auto& name = simulation.particles_[i]->parameters.sort_name;
    add(24, "dK_" + name, "{: .16e}", K_by_sort[i] - K0_by_sort[i]);
  }
  add(24, "dE", "{: .16e}", (w_E - w_E0));
  add(24, "dB", "{: .16e}", (w_B - w_B0));
  add(24, "dE+dB+dK", "{: .16e}", dF + (K - K0));
  add(24, "dMB", "{: .16e}", (a_MB - a_MB0));
  add(24, "dt * a_EJ", "{: .16e}", (dt * a_EJ));
  add(24, "dEB-dMB+dt*dEJ", "{: .16e}", dF - (a_MB - a_MB0) + dt * a_EJ);
  add(24, "dK-dMB+dt*dEJ", "{: .16e}", (K - K0) + (a_MB - a_MB0) - dt * a_EJ);
  add(24, "wK", "{: .16e}", (K));
  for (PetscInt i = 0; i < (PetscInt)K_by_sort.size(); ++i) {
    const auto& name = simulation.particles_[i]->parameters.sort_name;
    add(24, "wK_" + name, "{: .16e}", K_by_sort[i]);
  }
  add(24, "wE", "{: .16e}", (w_E));
  add(24, "wB", "{: .16e}", (w_B));
  add(24, "wEB + wK", "{: .16e}", w_E + w_B + K);

  const PetscReal dWE = w_E - w_E0;
  const PetscReal dWB = w_B - w_B0;

  Vec rotM_vec = nullptr;
  PetscReal a_EM = 0.0;

  PetscCall(DMGetGlobalVector(simulation.da, &rotM_vec));
  PetscCall(MatMult(simulation.rotM, simulation.M, rotM_vec));
  PetscCall(VecDot(simulation.E_hk, rotM_vec, &a_EM));
  PetscCall(DMRestoreGlobalVector(simulation.da, &rotM_vec));

  const PetscReal field_balance_direct = dWE + dWB + dt * (a_EJ + a_EM);

  add(24, "dWE", "{: .16e}", dWE);
  add(24, "dWB", "{: .16e}", dWB);
  add(24, "dt*EJ", "{: .16e}", dt * a_EJ);
  add(24, "dt*ErotM", "{: .16e}", dt * a_EM);
  add(24, "dWE+dWB+dt*(EJ+ErotM)", "{: .16e}", field_balance_direct);

  Vec sum = nullptr;
  Vec diff = nullptr;
  PetscReal norm[2];

  PetscCall(DMGetGlobalVector(charge_da, &diff));
  PetscCall(DMGetGlobalVector(charge_da, &sum));
  PetscCall(VecSet(sum, 0.0));

  // add_separator();

  PetscInt i = 0;
  for (; i < (PetscInt)charge_fields.size(); ++i) {
    PetscCall(VecCopy(charge_fields[i], diff));
    PetscCall(collect_charge_density(i));
    PetscCall(VecAYPX(diff, -1.0, charge_fields[i]));
    PetscCall(VecScale(diff, 1.0 / dt));

    PetscCall(VecAXPY(sum, 1.0, diff));

    PetscCall(MatMultAdd(divE, current_densities[i], diff, diff));
    PetscCall(VecNorm(diff, NORM_1_AND_2, norm));

    const auto& name = simulation.particles_[i]->parameters.sort_name;
    add(13, "N1dQ_" + name, "{: .6e}", norm[0]);
    add(13, "N2dQ_" + name, "{: .6e}", norm[1]);
  }

  PetscCall(MatMultAdd(divE, current_densities[i], sum, sum));
  PetscCall(VecNorm(sum, NORM_1_AND_2, norm));

  add(13, "N1dQ_tot", "{: .6e}", norm[0]);
  add(13, "N2dQ_tot", "{: .6e}", norm[1]);

  PetscCall(DMRestoreGlobalVector(charge_da, &sum));
  PetscCall(DMRestoreGlobalVector(charge_da, &diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace drift_kinetic
