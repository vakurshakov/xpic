#include "simulation_backup.h"

#include "src/utils/configuration.h"
#include "src/utils/mpi_binary_file.h"

SimulationBackup::SimulationBackup(const std::string& out_dir,
  PetscInt diagnose_period, std::map<std::string, Vec> fields,
  std::map<std::string, interfaces::Particles*> particles)
  : Diagnostic(out_dir, diagnose_period), fields_(fields), particles_(particles)
{
}


PetscErrorCode SimulationBackup::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(save(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SimulationBackup::execute(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(load(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode SimulationBackup::save(PetscInt t) const
{
  PetscFunctionBeginUser;
  if (t % diagnose_period_ == 0) {
    PetscCall(save_fields(t));
    PetscCall(save_particles(t));
    PetscCall(save_temporal_diagnostics(t));

    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    if (rank == 0) {
      t -= num_periods_being_kept * diagnose_period_;
      std::filesystem::remove_all(std::format("{}/{}", out_dir_, t));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SimulationBackup::save_fields(PetscInt t) const
{
  PetscFunctionBeginUser;
  PetscViewer viewer;

  for (const auto& [name, field] : fields_) {
    std::filesystem::path fname = std::format("{}/{}/{}", out_dir_, t, name);
    PetscCall(create_viewer(fname, PETSC_FALSE, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(field, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SimulationBackup::save_particles(PetscInt t) const
{
  PetscFunctionBeginUser;
  PetscViewer viewer;

  for (const auto& [name, container] : particles_) {
    std::filesystem::path fname = std::format("{}/{}/{}", out_dir_, t, name);

    PetscInt numparts = 0;
    for (const auto& cell : container->storage)
      numparts += (PetscInt)cell.size();
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &numparts, 1, MPIU_INT, MPIU_SUM, PETSC_COMM_WORLD));

    PetscCall(create_viewer(fname.string() + ".numparts", PETSC_TRUE, FILE_MODE_WRITE, &viewer));
    PetscCall(PetscViewerBinaryWrite(viewer, &numparts, 1, PETSC_INT));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(create_viewer(fname, PETSC_TRUE, FILE_MODE_WRITE, &viewer));
    for (const auto& cell : container->storage) {
      std::vector<Point> copy(cell.begin(), cell.end());
      PetscCall(PetscViewerBinaryWriteAll(viewer, copy.data(), 6 * (PetscInt)copy.size(), PETSC_DETERMINE, PETSC_DETERMINE, PETSC_REAL));
    }
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Copying `temporal` directory from simulation ouput to backup
PetscErrorCode SimulationBackup::save_temporal_diagnostics(PetscInt t) const
{
  PetscFunctionBeginUser;
  std::filesystem::path from = CONFIG().out_dir + "/temporal";
  std::filesystem::path to = std::format("{}/{}/temporal", out_dir_, t);
  PetscCall(saveload_temporal_diagnostics(from, to));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode SimulationBackup::load(PetscInt t)
{
  if (!std::filesystem::exists(out_dir_))
    throw std::runtime_error("Cannot load the timestep, no backup directory");

  PetscFunctionBeginUser;
  PetscCall(load_fields(t));
  PetscCall(load_particles(t));
  PetscCall(load_temporal_diagnostics(t));
  LOG("  Simulation is successfully loaded from {:.1f} [1/w_pe], {} [dt]", t * dt, t);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SimulationBackup::load_fields(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscViewer viewer;

  for (auto& [name, field] : fields_) {
    std::filesystem::path fname = std::format("{}/{}/{}", out_dir_, t, name);
    PetscCall(create_viewer(fname, PETSC_FALSE, FILE_MODE_READ, &viewer));
    PetscCall(VecLoad(field, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SimulationBackup::load_particles(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscViewer viewer;

  Point point;
  PetscInt numparts, count;

  for (auto& [name, container] : particles_) {
    std::filesystem::path fname = std::format("{}/{}/{}", out_dir_, t, name);

    PetscCall(create_viewer(fname.string() + ".numparts", PETSC_TRUE, FILE_MODE_READ, &viewer));
    PetscCall(PetscViewerBinaryRead(viewer, &numparts, 1, &count, PETSC_INT));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCheck(count == 1, PETSC_COMM_WORLD, PETSC_ERR_FILE_UNEXPECTED,
      "Incorrect number of particles to read is specified");

    PetscCall(create_viewer(fname, PETSC_TRUE, FILE_MODE_READ, &viewer));
    for (PetscInt i = 0; i < numparts; ++i) {
      PetscCall(PetscViewerBinaryRead(viewer, &point, 6, &count, PETSC_REAL));
      PetscCheck(count == 6, PETSC_COMM_WORLD, PETSC_ERR_FILE_UNEXPECTED,
        "Point structure consists of 6 PetscReal values, read: %" PetscInt_FMT, count);
      PetscCall(container->add_particle(point));
    }
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SimulationBackup::load_temporal_diagnostics(PetscInt t)
{
  PetscFunctionBeginUser;
  std::filesystem::path from = std::format("{}/{}/temporal", out_dir_, t);
  std::filesystem::path to = CONFIG().out_dir + "/temporal";
  PetscCall(saveload_temporal_diagnostics(from, to));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode SimulationBackup::create_viewer(const std::filesystem::path& fname,
  PetscBool skip, PetscFileMode mode, PetscViewer* viewer) const
{
  PetscFunctionBeginUser;
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  if (rank == 0 && mode != FILE_MODE_READ)
    std::filesystem::create_directories(fname.parent_path());

  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, viewer));
  PetscCall(PetscViewerSetType(*viewer, PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(*viewer, mode));
  PetscCall(PetscViewerBinarySetUseMPIIO(*viewer, PETSC_TRUE));
  PetscCall(PetscViewerBinarySetSkipInfo(*viewer, skip));
  PetscCall(PetscViewerBinarySetSkipHeader(*viewer, skip));
  PetscCall(PetscViewerFileSetName(*viewer, fname.c_str()));
  PetscCall(PetscViewerSetUp(*viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SimulationBackup::saveload_temporal_diagnostics(
  const std::filesystem::path& from, const std::filesystem::path& to) const
{
  if (!std::filesystem::exists(from))
    return PETSC_SUCCESS;

  PetscFunctionBeginUser;
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  if (rank == 0) {
    std::filesystem::copy(from, to,
      std::filesystem::copy_options::overwrite_existing |
        std::filesystem::copy_options::recursive);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
