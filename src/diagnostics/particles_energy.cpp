#include "particles_energy.h"

#include "src/utils/utils.h"

ParticlesEnergy::ParticlesEnergy(ParticlesPointersVector particles)
  : particles_(std::move(particles))
{
  std::fill_n(std::back_inserter(energies_), particles_.size(), 0);
}


ParticlesEnergy::ParticlesEnergy(
  const std::string& out_dir, ParticlesPointersVector particles)
  : interfaces::Diagnostic(out_dir),
    file_(SyncBinaryFile(out_dir_ + "/fields_energy.bin")),
    particles_(std::move(particles))
{
}

PetscErrorCode ParticlesEnergy::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(calculate_energies());

  PetscReal total = 0.0;

  for (const auto& energy : energies_) {
    PetscCall(file_.write_floats(3, energy.data));
    total += energy.elements_sum();
  }

  PetscCall(file_.write_floats(1, &total));

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
