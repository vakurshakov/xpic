#include "particles_energy.h"

#include "src/utils/utils.h"

ParticlesEnergy::ParticlesEnergy(ParticlesPointersVector particles)
  : particles_(std::move(particles))
{
  for (const auto& _ : particles_)
    energies_.emplace_back(Vector3R{});
}


ParticlesEnergy::ParticlesEnergy(
  const std::string& out_dir, ParticlesPointersVector particles)
  : interfaces::Diagnostic(out_dir),
    file_(SyncBinaryFile(out_dir_ + "/fields_energy.bin")),
    particles_(std::move(particles))
{
}

PetscErrorCode ParticlesEnergy::diagnose(timestep_t t)
{
  PetscFunctionBeginUser;
  PetscCall(calculate_energies());

  PetscReal total = 0.0;

  for (const auto& energy : energies_) {
    PetscCall(file_.write_floats(3, energy.data.data()));
    total += energy.elements_sum();
  }

  PetscCall(file_.write_floats(1, &total));

  if (t % diagnose_period == 0)
    file_.flush();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticlesEnergy::calculate_energies()
{
  auto calculate = [&](const interfaces::Particles* particles, Vector3R& result) {
    const PetscInt Np = particles->parameters.Np;
    const PetscReal m = particles->parameters.m;

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

    result = 0.5 * m * Vector3R{wx, wy, wz} / Np;
  };

  PetscFunctionBeginUser;
  for (std::size_t i = 0; i < particles_.size(); ++i) {
    energies_[i] = 0.0;
    calculate(particles_[i], energies_[i]);
  }

  std::vector<PetscReal> sendbuf(3 * energies_.size());
  std::vector<PetscReal> recvbuf(3 * energies_.size());

  for (std::size_t i = 0; i < energies_.size(); ++i) {
    sendbuf[i * 3 + 0] = energies_[i][X];
    sendbuf[i * 3 + 1] = energies_[i][Y];
    sendbuf[i * 3 + 2] = energies_[i][Z];
  }
  PetscCallMPI(MPI_Reduce(sendbuf.data(), recvbuf.data(), energies_.size(), MPIU_REAL, MPI_SUM, 0, PETSC_COMM_WORLD));

  for (std::size_t i = 0; i < recvbuf.size(); i += 3) {
    energies_[i / 3][X] = sendbuf[i + 0];
    energies_[i / 3][Y] = sendbuf[i + 1];
    energies_[i / 3][Z] = sendbuf[i + 2];
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
