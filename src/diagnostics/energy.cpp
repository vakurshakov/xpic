#include "energy.h"

#include "src/commands/fields_damping.h"
#include "src/commands/inject_particles.h"
#include "src/commands/remove_particles.h"
#include "src/utils/configuration.h"
#include "src/utils/utils.h"

Energy::Energy(const interfaces::Simulation& simulation)
  : simulation(simulation),
    energy(CONFIG().out_dir + "/temporal/energy.txt"),
    energy_cons(CONFIG().out_dir + "/temporal/energy_conservation.txt")
{
  auto& particles = simulation.particles_;
  std::fill_n(std::back_inserter(K), particles.size(), 0);
  std::fill_n(std::back_inserter(K0), particles.size(), 0);
  std::fill_n(std::back_inserter(std_K), particles.size(), 0);
}

PetscErrorCode Energy::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t == 0) {
    PetscCall(calculate_field());
    PetscCall(calculate_kinetic());
  }

  E0 = E;
  B0 = B;
  K0 = K;

  PetscCall(calculate_field());
  PetscCall(calculate_kinetic());

  PetscCall(fill_energy(t));
  PetscCall(fill_energy_cons(t));

  PetscCall(energy.diagnose(t));
  PetscCall(energy_cons.diagnose(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Energy::calculate_field()
{
  PetscFunctionBeginUser;
  PetscCall(VecNorm(simulation.E, NORM_2, &E));
  PetscCall(VecNorm(simulation.B, NORM_2, &B));
  E = 0.5 * POW2(E);
  B = 0.5 * POW2(B);

  Vector3R mean_E, mean_B;
  PetscCall(VecStrideSumAll(simulation.E, mean_E));
  PetscCall(VecStrideSumAll(simulation.B, mean_B));

  PetscReal g3 = (geom_nx * geom_ny * geom_nz);
  std_E = sqrt((E - 0.5 * mean_E.squared() / g3) / g3);
  std_B = sqrt((B - 0.5 * mean_B.squared() / g3) / g3);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Energy::calculate_kinetic()
{
  PetscFunctionBeginUser;
  PetscReal frac, m, mpw, vx, vy, vz, w;
  PetscInt n;

  for (std::size_t i = 0; i < simulation.particles_.size(); ++i) {
    auto& sort = simulation.particles_[i];

    m = sort->parameters.m;
    mpw = sort->parameters.n / (PetscReal)sort->parameters.Np;

    frac = 0.5 * m * mpw;
    vx = vy = vz = w = 0;
    n = 0;

#pragma omp parallel for reduction(+ : vx, vy, vz, w, n)
    for (auto&& cell : sort->storage) {
      for (auto&& point : cell) {
        vx += point.px();
        vy += point.py();
        vz += point.pz();

        w += point.p.squared();
        n++;
      }
    }

    K[i] = frac * w;

    PetscReal v[4] = {vx, vy, vz, w};
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, v, 4, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

    if (n == 0) {
      K[i] = 0;
      std_K[i] = 0;
      continue;
    }

    PetscReal s = v[3] - (v[X] * v[X] + v[Y] * v[Y] + v[Z] * v[Z]) / n;
    std_K[i] = frac * sqrt(abs(s) / n);
  }

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, K.data(), K.size(), MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Energy::fill_energy(PetscInt t)
{
  PetscFunctionBeginUser;
  energy.add(6, "Time", "{:d}", t);

  auto& particles = simulation.particles_;
  PetscInt i, size = (PetscInt)particles.size();

  energy.add(13, "wE", "{: .6e}", E);
  energy.add(13, "wB", "{: .6e}", B);

  for (i = 0; i < size; ++i) {
    auto& name = particles[i]->parameters.sort_name;
    energy.add(13, "wK_" + name, "{: .6e}", K[i]);
  }

  energy.add(13, "sE", "{: .6e}", std_E);
  energy.add(13, "sB", "{: .6e}", std_B);

  for (i = 0; i < size; ++i) {
    auto& name = particles[i]->parameters.sort_name;
    energy.add(13, "sK_" + name, "{: .6e}", std_K[i]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Energy::fill_energy_cons(PetscInt t)
{
  PetscFunctionBeginUser;
  energy_cons.add(6, "Time", "{:d}", t);

  dE = E - E0;
  dB = B - B0;
  K = K;

  dF = dE + dB;
  energy_cons.add(13, "dE", "{: .6e}", dE);
  energy_cons.add(13, "dB", "{: .6e}", dB);

  dK = 0;
  for (PetscInt i = 0; i < (PetscInt)K.size(); ++i) {
    auto&& n = simulation.particles_[i]->parameters.sort_name;
    energy_cons.add(13, "dK_" + n, "{: .6e}", K[i] - K0[i]);
    dK += K[i] - K0[i];
  }

  for (const auto& command : simulation.step_presets_) {
    if (auto&& damp = dynamic_cast<FieldsDamping*>(command.get())) {
      energy_cons.add(13, "Damped(E+B)", "{: .6e}", damp->get_damped_energy());
      dF += damp->get_damped_energy();
    }
    if (auto&& injection = dynamic_cast<InjectParticles*>(command.get())) {
      auto&& ni = injection->get_ionized_name();
      auto&& ne = injection->get_ejected_name();
      auto&& wi = injection->get_ionized_energy();
      auto&& we = injection->get_ejected_energy();
      energy_cons.add(13, "Inj_" + ni, "{: .6e}", wi);
      energy_cons.add(13, "Inj_" + ne, "{: .6e}", we);
      dK -= wi + we;
    }
    if (auto&& remove = dynamic_cast<RemoveParticles*>(command.get())) {
      auto&& n = remove->get_particles_name();
      auto&& w = remove->get_removed_energy();
      energy_cons.add(13, "Rm_" + n, "{: .6e}", w);
      dK += remove->get_removed_energy();
    }
  }

  energy_cons.add(13, "dE+dB+dK", "{: .6e}", dF + dK);
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscReal Energy::get_field(const Vector3R& f)
{
  return 0.5 * f.squared();
}

PetscReal Energy::get_kinetic(const Vector3R& p, PetscReal m, PetscReal mpw)
{
  return 0.5 * (m * p.squared()) * mpw;
}
