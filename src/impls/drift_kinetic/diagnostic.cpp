#include "diagnostic.h"

#include "src/algorithms/implicit_drift_kinetic.h"
#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "src/utils/operators.h"
#include "src/utils/shape.h"
#include "src/utils/utils.h"

namespace drift_kinetic {

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

inline std::vector<PetscReal> dk_get_momentum_flux(
  const Particles& particles, const PointByField& point, PetscReal)
{
  return momentum_flux(particles.parameters.m, point.p);
}

inline std::vector<PetscReal> dk_get_momentum_flux_diag(
  const Particles& particles, const PointByField& point, PetscReal)
{
  return momentum_flux_diag(particles.parameters.m, point.p);
}

inline std::vector<PetscReal> dk_get_momentum_flux_cyl(
  const Particles& particles, const PointByField& point, PetscReal)
{
  return momentum_flux_cyl(particles.parameters.m, point.r, point.p);
}

inline std::vector<PetscReal> dk_get_momentum_flux_diag_cyl(
  const Particles& particles, const PointByField& point, PetscReal)
{
  return momentum_flux_diag_cyl(particles.parameters.m, point.r, point.p);
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

std::unique_ptr<DkDistributionMoment> DkDistributionMoment::create(
  const std::string& out_dir, const Particles& particles,  //
  const Moment& moment, const Region& region)
{
  PetscFunctionBeginUser;
  MPI_Comm newcomm;
  PetscCallAbort(PETSC_COMM_WORLD, World::create_local_comm(particles.world.da, region, &newcomm));

  auto* diagnostic =
    new DkDistributionMoment(out_dir, particles, moment, newcomm);
  PetscCallAbort(PETSC_COMM_WORLD, diagnostic->set_data_views(region));
  PetscFunctionReturn(std::unique_ptr<DkDistributionMoment>(diagnostic));
}

std::unique_ptr<DkDistributionMoment> DkDistributionMoment::create(
  const std::string& out_dir, const Particles& particles,  //
  const DkMoment& moment, const Region& region)
{
  PetscFunctionBeginUser;
  MPI_Comm newcomm;
  PetscCallAbort(PETSC_COMM_WORLD, World::create_local_comm(particles.world.da, region, &newcomm));

  auto* diagnostic =
    new DkDistributionMoment(out_dir, particles, moment, newcomm);
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
  ParticleStorage::Attributes f;
  PetscCall(dk_particles.storage.get_attributes(f));
  const DkPointArrays arrays(f);
  const std::vector<PetscInt>& offsets = dk_particles.storage.cell_offsets();

  // DK moments receive |B| interpolated at the gyrocenter; check out the
  // current B field once before the particle loop. (The non-DK `moment`
  // path never touches B.)
  Arr B_arr_local = nullptr;
  const auto& sim = dk_particles.simulation();
  if (dk_moment) {
    PetscCall(DMGlobalToLocal(sim.da, sim.B, INSERT_VALUES, sim.B_loc));
    PetscCall(DMDAVecGetArrayRead(sim.da, sim.B_loc, &B_arr_local));
  }

  PetscCall(::DistributionMoment::collect([&](PetscInt g, auto&& deposit)
  {
    for (PetscInt i = offsets[g]; i < offsets[g + 1]; i++) {
      const PointByField point = arrays.load(i);
      if (!dk_moment) {
        deposit(point.r, moment(particles, ::Point(point.r, point.p)));
        continue;
      }

      // Interpolate |B| at the gyrocenter with the 2nd-order spline.
      Vector3R B_p;
      DriftKineticEsirkepov interp(dk_particles.world, B_arr_local);
      interp.interpolate_B(B_p, point.r);

      deposit(point.r, dk_moment(dk_particles, point, B_p.length()));
    }
  }));

  if (dk_moment) {
    PetscCall(DMDAVecRestoreArrayRead(sim.da, sim.B_loc, &B_arr_local));
  }
  PetscCall(dk_particles.storage.restore_attributes(f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

std::unique_ptr<DkDistributionFunction> DkDistributionFunction::create(
  const std::string& out_dir, const Particles& particles,
  const DkPhaseAxis& v_parallel, const DkPhaseAxis& mu_p,
  PetscInt diagnose_period, PetscInt max_frames)
{
  PetscFunctionBeginUser;
  MPI_Comm comm = MPI_COMM_NULL;
  PetscCallAbort(PETSC_COMM_WORLD, MPI_Comm_dup(PETSC_COMM_WORLD, &comm));

  auto* diagnostic = new DkDistributionFunction(
    out_dir, particles, v_parallel, mu_p, diagnose_period, max_frames, comm);
  PetscCallAbort(PETSC_COMM_WORLD, diagnostic->set_data_views({}));
  PetscFunctionReturn(std::unique_ptr<DkDistributionFunction>(diagnostic));
}

DkDistributionFunction::DkDistributionFunction(const std::string& out_dir,
  const Particles& particles, const DkPhaseAxis& v_parallel,
  const DkPhaseAxis& mu_p, PetscInt diagnose_period, PetscInt max_frames,
  MPI_Comm comm)
  : FieldView(out_dir, particles.world.da, nullptr, comm),
    particles(particles),
    v_parallel(v_parallel),
    mu_p(mu_p),
    max_frames(max_frames)
{
  diagnose_period_ = diagnose_period;
}

PetscErrorCode DkDistributionFunction::set_data_views(const Region& /* reg */)
{
  PetscFunctionBeginUser;
  const Vector3I& start = particles.world.start;
  const Vector3I& size = particles.world.size;

  const PetscInt nv = v_parallel.bins;
  const PetscInt nmu = mu_p.bins;

  const PetscInt global_sizes[5]{geom_nz, geom_ny, geom_nx, nv, nmu};
  const PetscInt local_sizes[5]{size[Z], size[Y], size[X], nv, nmu};
  const PetscInt file_starts[5]{start[Z], start[Y], start[X], 0, 0};
  const PetscInt mem_starts[5]{0, 0, 0, 0, 0};

  PetscCall(create_subarray(5, local_sizes, local_sizes, mem_starts, &memview));
  PetscCall(create_subarray(5, global_sizes, local_sizes, file_starts, &fileview));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DkDistributionFunction::diagnose(PetscInt t)
{
  if (t % diagnose_period_ != 0 ||
    (max_frames > 0 && frames_written >= max_frames))
    return PETSC_SUCCESS;

  PetscFunctionBeginUser;
  PetscInt nv = v_parallel.bins;
  PetscInt nmu = mu_p.bins;
  PetscReal dv = (v_parallel.max - v_parallel.min) / nv;
  PetscReal dmu = (mu_p.max - mu_p.min) / nmu;
  PetscReal histogram_weight = particles.parameters.mpw / (dv * dmu);

  PetscInt coord_cells = particles.world.size.elements_product();
  PetscInt velocity_cells = nv * nmu;

  std::vector<PetscReal> histogram(coord_cells * velocity_cells, 0.0f);

  PetscInt included_local = 0;
  PetscInt omitted_local = 0;
  ParticleStorage::Attributes f;
  PetscCall(particles.storage.get_attributes(f));
  const DkPointArrays arrays(f);
  const std::vector<PetscInt>& offsets = particles.storage.cell_offsets();

#pragma omp parallel for reduction(+ : included_local, omitted_local)
  for (PetscInt g = 0; g < coord_cells; ++g) {
    for (PetscInt n = offsets[g]; n < offsets[g + 1]; ++n) {
      const PointByField point = arrays.load(n);
      PetscInt iv = FLOOR_STEP(point.p_parallel - v_parallel.min, dv);
      PetscInt imu = FLOOR_STEP(point.mu_p - mu_p.min, dmu);

      if (iv < 0 || iv >= nv || imu < 0 || imu >= nmu) {
        omitted_local++;
        continue;
      }

      included_local++;
      PetscInt index = (g * nv + iv) * nmu + imu;

#pragma omp atomic update
      histogram[index] += histogram_weight;
    }
  }
  PetscCall(particles.storage.restore_attributes(f));

  PetscInt counts[2]{included_local, omitted_local};
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPIU_INT, MPI_SUM, comm));

  PetscInt total = counts[0] + counts[1];
  PetscReal omitted_fraction = total > 0 ? (PetscReal)counts[1] / total : 0;
  LOG("  DkDistributionFunction {}:", particles.parameters.sort_name);
  LOG("    included {}, omitted {}, fraction {:.6e}", counts[0], counts[1], omitted_fraction);

  PetscCall(open(out_dir_ + "/" + format_time(t)));
  PetscCall(write(histogram.size(), histogram.data()));
  PetscCall(close());
  frames_written++;
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
  per_sort.assign(simulation.particles_.size(), 0);
  total = 0;

  for (PetscInt i = 0; i < (PetscInt)simulation.particles_.size(); ++i)
    per_sort[i] = simulation.particles_[i]->kinetic_energy_local();

  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(MPI_IN_PLACE, per_sort.data(), (PetscMPIInt)per_sort.size(), MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

  for (PetscReal kinetic : per_sort)
    total += kinetic;
}

PetscErrorCode EnergyConservation::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (!initialized)
    calculate_kinetic_energies(K_by_sort, K);

  K0 = K;
  K0_by_sort = K_by_sort;
  PetscCall(TableDiagnostic::diagnose(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(TableDiagnostic::initialize());
  PetscCall(DMCreateGlobalVector(simulation.da, &E_prev));
  PetscCall(DMCreateGlobalVector(simulation.da, &B_prev));
  PetscCall(VecCopy(simulation.E, E_prev));
  PetscCall(VecCopy(simulation.B, B_prev));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&E_prev));
  PetscCall(VecDestroy(&B_prev));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  add(6, "Time", "{:d}", t);
  for (const auto& sort : simulation.particles_) {
    const auto& name = sort->parameters.sort_name;
    add(13, "MaxPushIt_" + name, "{:d}", sort->get_max_iteration_number());
    const std::string retries_title = "PushRetries_" + name;
    const std::string failures_title = "PushLeafFailures_" + name;
    add(std::max<PetscInt>(13, static_cast<PetscInt>(retries_title.size())),
      retries_title, "{:d}", sort->get_push_retries());
    add(std::max<PetscInt>(13, static_cast<PetscInt>(failures_title.size())),
      failures_title, "{:d}", sort->get_push_leaf_failures());
  }
  add(13, "AvgFieldIt", "{:d}", simulation.snes_iteration_number);

  PetscReal w_E = 0;
  PetscReal w_B = 0;
  PetscReal a_EJ = 0;
  PetscReal a_MB = 0;
  PetscCall(VecDot(simulation.E, simulation.E, &w_E));
  PetscCall(VecDot(simulation.B, simulation.B, &w_B));
  PetscCall(VecDot(simulation.E_hk, simulation.J, &a_EJ));
  PetscCall(VecDot(simulation.M, simulation.B, &a_MB));
  w_E *= 0.5;
  w_B *= 0.5;
  calculate_kinetic_energies(K_by_sort, K);

  Vec fsum = nullptr;
  Vec fdiff = nullptr;
  PetscCall(DMGetGlobalVector(simulation.da, &fsum));
  PetscCall(DMGetGlobalVector(simulation.da, &fdiff));

  PetscReal dWE = 0;
  PetscReal dWB = 0;
  PetscCall(VecWAXPY(fdiff, -1, E_prev, simulation.E));
  PetscCall(VecWAXPY(fsum, +1, E_prev, simulation.E));
  PetscCall(VecDot(fsum, fdiff, &dWE));
  dWE *= 0.5;

  PetscCall(VecWAXPY(fdiff, -1, B_prev, simulation.B));
  PetscCall(VecWAXPY(fsum, +1, B_prev, simulation.B));
  PetscCall(VecDot(fsum, fdiff, &dWB));
  dWB *= 0.5;

  PetscCall(VecCopy(simulation.E, E_prev));
  PetscCall(VecCopy(simulation.B, B_prev));

  PetscCall(DMRestoreGlobalVector(simulation.da, &fsum));
  PetscCall(DMRestoreGlobalVector(simulation.da, &fdiff));

  add(13, "dK", "{: .6e}", K - K0);
  for (PetscInt i = 0; i < (PetscInt)K_by_sort.size(); ++i) {
    const auto& name = simulation.particles_[i]->parameters.sort_name;
    add(24, "dK_" + name, "{: .6e}", K_by_sort[i] - K0_by_sort[i]);
  }
  add(13, "dE", "{: .6e}", dWE);
  add(13, "dB", "{: .6e}", dWB);
  add(13, "dE+dB+dK", "{: .6e}", dWE + dWB + (K - K0));
  add(13, "dK-dMB+dt*dEJ", "{: .6e}",
    (K - K0) + (a_MB - simulation.a_MB0) - dt * a_EJ);

  add(13, "wK", "{: .6e}", K);
  for (PetscInt i = 0; i < (PetscInt)K_by_sort.size(); ++i) {
    const auto& name = simulation.particles_[i]->parameters.sort_name;
    add(13, "wK_" + name, "{: .6e}", K_by_sort[i]);
  }
  add(13, "wE", "{: .6e}", w_E);
  add(13, "wB", "{: .6e}", w_B);
  add(13, "wEB+wK", "{: .6e}", w_E + w_B + K);
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace drift_kinetic
