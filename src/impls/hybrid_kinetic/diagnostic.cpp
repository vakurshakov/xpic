#include "diagnostic.h"

#include "src/impls/hybrid_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "src/utils/shape.h"

namespace hybrid_kinetic {

namespace {

// Parabolic kernel at integer nodes, the density shape of both pushers.
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
      cache[i] =
        sfunc(p_r[X] - g_x) * sfunc(p_r[Y] - g_y) * sfunc(p_r[Z] - g_z);
    }
  }
};

PetscErrorCode collect_charge_density(
  DM da, const ParticleStorage& storage, PetscReal q_mpw, Vec rho)
{
  PetscFunctionBeginUser;
  Vec local;
  PetscCall(DMGetLocalVector(da, &local));
  PetscCall(VecSet(local, 0));
  PetscCall(VecSet(rho, 0));

  PetscReal*** arr;
  PetscCall(DMDAVecGetArrayWrite(da, local, &arr));

  ParticleStorage::Attributes f;
  PetscCall(storage.get_attribute(f, "r"));
  const PetscReal* r = f["r"];
  ChargeShape shape;

#pragma omp parallel for private(shape)
  for (PetscInt p = 0; p < f.size; p++) {
    shape.setup(Vector3R(r + 3 * p));

    for (PetscInt i = 0; i < shape.shm; i++) {
      const PetscInt x = shape.start[X] + i % shape.shw;
      const PetscInt y = shape.start[Y] + (i / shape.shw) % shape.shw;
      const PetscInt z = shape.start[Z] + (i / shape.shw) / shape.shw;

#pragma omp atomic update
      arr[z][y][x] += q_mpw * shape.cache[i];
    }
  }

  PetscCall(storage.restore_attributes(f));
  PetscCall(DMDAVecRestoreArrayWrite(da, local, &arr));
  PetscCall(DMLocalToGlobal(da, local, ADD_VALUES, rho));
  PetscCall(DMRestoreLocalVector(da, &local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscInt width(const std::string& title)
{
  return std::max<PetscInt>(13, (PetscInt)title.size());
}

}  // namespace

EnergyConservation::EnergyConservation(const Simulation& simulation)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/hybrid_diagnostic.txt"),
    simulation(simulation)
{
}

// Called at the first diagnosed step, also when the run starts from a backup.
PetscErrorCode EnergyConservation::initialize()
{
  PetscFunctionBeginUser;
  if (initialized)
    PetscFunctionReturn(PETSC_SUCCESS);

  DM da = simulation.da_rho;

  for (const auto& ptr : simulation.particles_) {
    const drift_kinetic::Particles& sort = *ptr;
    sorts.emplace_back(SortView{
      sort,
      [&sort] { return sort.kinetic_energy_local(); },
      [&sort, da](Vec rho)
      {
        return collect_charge_density(da, sort.storage,
          sort.parameters.q * sort.parameters.mpw, rho);
      },
    });
  }

  for (const auto& ptr : simulation.kinetic_particles_) {
    const KineticParticles& sort = *ptr;
    sorts.emplace_back(SortView{
      sort,
      [&sort] { return sort.kinetic_energy_local(); },
      [&sort, da](Vec rho)
      {
        return collect_charge_density(da, sort.storage,
          sort.parameters.q * sort.parameters.mpw, rho);
      },
    });
  }

  rho0.resize(sorts.size());
  for (std::size_t i = 0; i < sorts.size(); i++) {
    PetscCall(DMCreateGlobalVector(da, &rho0[i]));
    PetscCall(sorts[i].collect_charge_density(rho0[i]));
  }

  PetscCall(calculate_kinetic_energies(K0));
  PetscCall(VecDot(simulation.E, simulation.E, &w_E0));
  PetscCall(VecDot(simulation.B, simulation.B, &w_B0));
  w_E0 *= 0.5;
  w_B0 *= 0.5;

  initialized = true;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::finalize()
{
  PetscFunctionBeginUser;
  for (auto& rho : rho0)
    PetscCall(VecDestroy(&rho));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(initialize());

  add(6, "Time", "{:d}", t);
  add(13, "AvgFieldIt", "{:d}", simulation.snes_iteration_number);

  std::vector<PetscReal> K;
  PetscCall(calculate_kinetic_energies(K));

  PetscReal w_E, w_B;
  PetscCall(VecDot(simulation.E, simulation.E, &w_E));
  PetscCall(VecDot(simulation.B, simulation.B, &w_B));
  w_E *= 0.5;
  w_B *= 0.5;

  PetscReal w_K = 0;
  PetscReal dK = 0;
  for (std::size_t i = 0; i < sorts.size(); i++) {
    w_K += K[i];
    dK += K[i] - K0[i];
  }

  add(13, "dK", "{: .6e}", dK);
  for (std::size_t i = 0; i < sorts.size(); i++) {
    const std::string title = "dK_" + sorts[i].sort.parameters.sort_name;
    add(width(title), title, "{: .6e}", K[i] - K0[i]);
  }
  add(13, "dE", "{: .6e}", w_E - w_E0);
  add(13, "dB", "{: .6e}", w_B - w_B0);
  add(13, "dE+dB+dK", "{: .6e}", (w_E - w_E0) + (w_B - w_B0) + dK);

  add(13, "wK", "{: .6e}", w_K);
  for (std::size_t i = 0; i < sorts.size(); i++) {
    const std::string title = "wK_" + sorts[i].sort.parameters.sort_name;
    add(width(title), title, "{: .6e}", K[i]);
  }
  add(13, "wE", "{: .6e}", w_E);
  add(13, "wB", "{: .6e}", w_B);
  add(13, "wEB + wK", "{: .6e}", w_E + w_B + w_K);

  // Split of the balance: the field part dW_EB + dt E.J - M.dB and, for each
  // sort, dK - dt E.J_s + M_s.dB (M_s = 0 for kinetic sorts). The total error
  // dE+dB+dK is their sum, so these columns localize it.
  Vec dB;
  PetscReal EJ, MdB;
  PetscCall(DMGetGlobalVector(simulation.da, &dB));
  PetscCall(VecWAXPY(dB, -1, simulation.B_hk, simulation.B));
  PetscCall(VecScale(dB, 2));

  PetscCall(VecDot(simulation.E_hk, simulation.J, &EJ));
  PetscCall(VecDot(simulation.M, dB, &MdB));
  add(width("dW-tEJ_field"), "dW-tEJ_field", "{: .6e}",
    (w_E - w_E0) + (w_B - w_B0) + dt * EJ - MdB);

  std::size_t i = 0;
  for (const auto& sort : simulation.particles_) {
    PetscCall(VecDot(simulation.E_hk, sort->J, &EJ));
    PetscCall(VecDot(sort->M, dB, &MdB));

    const std::string title = "dK-tEJ_" + sort->parameters.sort_name;
    add(width(title), title, "{: .6e}", K[i] - K0[i] - dt * EJ + MdB);
    i++;
  }

  for (const auto& sort : simulation.kinetic_particles_) {
    PetscCall(VecDot(simulation.E_hk, sort->J, &EJ));

    const std::string title = "dK-tEJ_" + sort->parameters.sort_name;
    add(width(title), title, "{: .6e}", K[i] - K0[i] - dt * EJ);
    i++;
  }
  PetscCall(DMRestoreGlobalVector(simulation.da, &dB));

  K0 = K;
  w_E0 = w_E;
  w_B0 = w_B;

  // Continuity residual (rho^{n+1} - rho^{n}) / dt + div(J^{n+1/2}),
  // `rho0` is turned into the residual and then replaced by `rho`.
  DM da = simulation.da_rho;
  Vec rho, sum;
  PetscReal norm[2];
  PetscCall(DMGetGlobalVector(da, &rho));
  PetscCall(DMGetGlobalVector(da, &sum));
  PetscCall(VecSet(sum, 0));

  for (std::size_t i = 0; i < sorts.size(); i++) {
    PetscCall(sorts[i].collect_charge_density(rho));
    PetscCall(VecAYPX(rho0[i], -1, rho));
    PetscCall(VecScale(rho0[i], 1 / dt));
    PetscCall(VecAXPY(sum, 1, rho0[i]));

    PetscCall(MatMultAdd(simulation.divE, sorts[i].sort.J, rho0[i], rho0[i]));
    PetscCall(VecNorm(rho0[i], NORM_1_AND_2, norm));
    PetscCall(VecCopy(rho, rho0[i]));

    const std::string& name = sorts[i].sort.parameters.sort_name;
    add(width("N1dQ_" + name), "N1dQ_" + name, "{: .6e}", norm[0]);
    add(width("N2dQ_" + name), "N2dQ_" + name, "{: .6e}", norm[1]);
  }

  PetscCall(MatMultAdd(simulation.divE, simulation.J, sum, sum));
  PetscCall(VecNorm(sum, NORM_1_AND_2, norm));

  add(13, "N1dQ_tot", "{: .6e}", norm[0]);
  add(13, "N2dQ_tot", "{: .6e}", norm[1]);

  PetscCall(DMRestoreGlobalVector(da, &rho));
  PetscCall(DMRestoreGlobalVector(da, &sum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::calculate_kinetic_energies(
  std::vector<PetscReal>& K) const
{
  PetscFunctionBeginUser;
  K.resize(sorts.size());
  for (std::size_t i = 0; i < sorts.size(); i++)
    K[i] = sorts[i].kinetic_energy_local();

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, K.data(), (PetscMPIInt)K.size(), MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace hybrid_kinetic
