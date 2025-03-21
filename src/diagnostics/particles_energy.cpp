#include "particles_energy.h"

#include "src/utils/utils.h"

ParticlesEnergy::ParticlesEnergy(
  std::vector<const interfaces::Particles*> particles)
  : particles_(std::move(particles))
{
  std::fill_n(std::back_inserter(energies_), particles_.size(), 0);
}


ParticlesEnergy::ParticlesEnergy(const std::string& out_dir,
  std::vector<const interfaces::Particles*> particles)
  : interfaces::Diagnostic(out_dir),
    file_(SyncFile(out_dir_ + "/temporal/particles_energy.txt")),
    particles_(std::move(particles))
{
  std::fill_n(std::back_inserter(energies_), particles_.size(), 0);
}

PetscErrorCode ParticlesEnergy::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t == 0) {
    for (const auto& particles : particles_) {
      const auto& name = particles->parameters.sort_name;
      file_() << std::format("{:>14s}  ", "Kx_" + name);
      file_() << std::format("{:>14s}  ", "Ky_" + name);
      file_() << std::format("{:>14s}  ", "Kz_" + name);
      file_() << std::format("{:>14s}  ", "K_" + name);
    }
    file_() << std::format("{:>14s}", "Total(K)") << "\n";
  }

  PetscCall(calculate_energies());

  PetscReal total = 0.0;
  for (const auto& energy : energies_) {
    file_() << std::format("{:.8e}  ", energy[X]);
    file_() << std::format("{:.8e}  ", energy[Y]);
    file_() << std::format("{:.8e}  ", energy[Z]);
    file_() << std::format("{:.8e}  ", energy.elements_sum());
    total += energy.elements_sum();
  }
  file_() << std::format("{:.8e}", total) << "\n";

  if (t % diagnose_period_ == 0)
    file_.flush();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticlesEnergy::calculate_energies()
{
  auto calculate = [&](const interfaces::Particles* particles, Vector3R& result) {
    const PetscReal m = particles->parameters.m;
    const PetscInt Np = particles->parameters.Np;

    PetscReal wx = 0.0;
    PetscReal wy = 0.0;
    PetscReal wz = 0.0;

#pragma omp parallel for reduction(+ : wx, wy, wz)
    for (auto&& cell : particles->storage) {
      for (auto&& point : cell) {
        wx += POW2(point.px());
        wy += POW2(point.py());
        wz += POW2(point.pz());
      }
    }

    result += 0.5 * m * Vector3R{wx, wy, wz} / Np;
  };

  PetscFunctionBeginUser;
  for (std::size_t i = 0; i < particles_.size(); ++i) {
    energies_[i] = 0.0;
    calculate(particles_[i], energies_[i]);
  }

  std::vector<PetscReal> buf(3 * energies_.size());

  for (std::size_t i = 0; i < energies_.size(); ++i) {
    buf[i * 3 + 0] = energies_[i][X];
    buf[i * 3 + 1] = energies_[i][Y];
    buf[i * 3 + 2] = energies_[i][Z];
  }
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, buf.data(), buf.size(), MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

  for (std::size_t i = 0; i < buf.size(); i += 3) {
    energies_[i / 3][X] = buf[i + 0];
    energies_[i / 3][Y] = buf[i + 1];
    energies_[i / 3][Z] = buf[i + 2];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

std::vector<PetscReal> ParticlesEnergy::get_energies() const
{
  std::vector<PetscReal> result;
  result.reserve(energies_.size());

  for (const auto& energy : energies_)
    result.emplace_back(energy.elements_sum());

  return result;
}

/* static */ PetscReal ParticlesEnergy::get(
  const Vector3R& p, PetscReal m, PetscInt Np)
{
  return 0.5 * (m * p.squared()) / Np;
}
