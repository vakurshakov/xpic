#include "energy.h"

#include "src/utils/configuration.h"
#include "src/utils/utils.h"

Energy::Energy(Vec E, Vec B, std::vector<const interfaces::Particles*> particles)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/energy.txt"),
    E(E),
    B(B),
    particles(particles)
{
  std::fill_n(std::back_inserter(w_K), particles.size(), 0.0);
  std::fill_n(std::back_inserter(std_K), particles.size(), 0.0);
}

PetscErrorCode Energy::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(calculate_energies());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Energy::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  add(6, "Time", "{:d}", t);

  PetscCall(calculate_energies());

  PetscInt i, size = (PetscInt)particles.size();

  add(13, "wE", "{: .6e}", w_E);
  add(13, "wB", "{: .6e}", w_B);
  for (i = 0; i < size; ++i)
    add(13, "wK_" + particles[i]->parameters.sort_name, "{: .6e}", w_K[i]);

  add(13, "sE", "{: .6e}", std_E);
  add(13, "sB", "{: .6e}", std_B);
  for (i = 0; i < size; ++i)
    add(13, "sK_" + particles[i]->parameters.sort_name, "{: .6e}", std_K[i]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Energy::calculate_energies()
{
  PetscFunctionBeginUser;
  PetscCall(VecNorm(E, NORM_2, &w_E));
  PetscCall(VecNorm(B, NORM_2, &w_B));
  w_E = 0.5 * POW2(w_E);
  w_B = 0.5 * POW2(w_B);

  Vector3R mean_E, mean_B;
  PetscCall(VecStrideSumAll(E, mean_E));
  PetscCall(VecStrideSumAll(B, mean_B));

  PetscReal g3 = (geom_nx * geom_ny * geom_nz);
  std_E = sqrt((w_E - 0.5 * mean_E.squared() / g3) / g3);
  std_B = sqrt((w_B - 0.5 * mean_B.squared() / g3) / g3);

  PetscReal frac, m, mpw, vx, vy, vz, w;
  PetscInt n;

  for (std::size_t i = 0; i < particles.size(); ++i) {
    auto&& sort = particles[i];

    m = sort->parameters.m;
    mpw = sort->parameters.n / (PetscReal)sort->parameters.Np;

    frac = 0.5 * m * mpw;
    vx = vy = vz = w = 0.0;
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

    w_K[i] = frac * w;

    PetscReal v[4] = {vx, vy, vz, w};
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, v, 4, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

    if (n == 0) {
      w_K[i] = 0;
      std_K[i] = 0;
      continue;
    }

    PetscReal s = v[3] - (v[X] * v[X] + v[Y] * v[Y] + v[Z] * v[Z]) / n;
    std_K[i] = frac * sqrt(abs(s) / n);
  }

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, w_K.data(), w_K.size(), MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal Energy::get_electric_energy() const
{
  return w_E;
}

PetscReal Energy::get_magnetic_energy() const
{
  return w_B;
}

std::vector<PetscReal> Energy::get_kinetic_energies() const
{
  return w_K;
}

PetscReal Energy::get_field(const Vector3R& f)
{
  return 0.5 * f.squared();
}

PetscReal Energy::get_kinetic(const Vector3R& p, PetscReal m, PetscInt Np)
{
  return 0.5 * (m * p.squared()) / Np;
}
