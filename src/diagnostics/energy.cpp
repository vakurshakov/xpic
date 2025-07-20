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
  std_E = std::sqrt((w_E - 0.5 * mean_E.squared() / g3) / g3);
  std_B = std::sqrt((w_B - 0.5 * mean_B.squared() / g3) / g3);

  for (std::size_t i = 0; i < particles.size(); ++i) {
    auto&& sort = particles[i];

    PetscReal frac = 0.5 * sort->parameters.m / sort->parameters.Np;
    PetscReal vx = 0.0, vy = 0.0, vz = 0.0, w = 0.0;
    PetscInt n = 0;

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
    std_K[i] =
      frac * std::sqrt(std::abs(w - Vector3R{vx, vy, vz}.squared() / n) / n);
  }

  std::vector<PetscReal> buf(2 * particles.size());

  for (std::size_t i = 0; i < particles.size(); ++i) {
    buf[i * 2 + 0] = w_K[i];
    buf[i * 2 + 1] = std_K[i];
  }
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, buf.data(), buf.size(), MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

  for (std::size_t i = 0; i < buf.size(); i += 2) {
    w_K[i / 2] = buf[i + 0];
    std_K[i / 2] = buf[i + 1];
  }
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
