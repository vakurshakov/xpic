#ifndef SRC_DIAGNOSTICS_SIMULATION_BACKUP_H
#define SRC_DIAGNOSTICS_SIMULATION_BACKUP_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/diagnostic.h"
#include "src/interfaces/particles.h"

/**
 * @brief This class is used to save and load simulation backups.
 * It is derived from two interfaces because of the following reasons:
 * `Diagnostic` - is used to embed its `save()` into common diagnostics pipeline.
 * `Command` - is used to alter the "Presets" behavior and call `load()`.
 */
class SimulationBackup : public interfaces::Diagnostic,
                         public interfaces::Command {
public:
  SimulationBackup(const std::string& out_dir,  //
    PetscInt diagnose_period, std::map<std::string, Vec> fields,
    std::map<std::string, interfaces::Particles*> particles);

  PetscErrorCode save(PetscInt t) const;
  PetscErrorCode load(PetscInt t);

private:
  PetscErrorCode diagnose(PetscInt t) override;
  PetscErrorCode execute(PetscInt t) override;

  PetscErrorCode save_fields(PetscInt t) const;
  PetscErrorCode save_particles(PetscInt t) const;
  PetscErrorCode save_temporal_diagnostics(PetscInt t) const;

  PetscErrorCode load_fields(PetscInt t);
  PetscErrorCode load_particles(PetscInt t);
  PetscErrorCode load_temporal_diagnostics(PetscInt t);

  PetscErrorCode create_viewer(const std::filesystem::path& name,
    PetscBool skip, PetscFileMode mode, PetscViewer* viewer) const;

  PetscErrorCode saveload_temporal_diagnostics(
    const std::filesystem::path& from, const std::filesystem::path& to) const;

  std::map<std::string, Vec> fields_;
  std::map<std::string, interfaces::Particles*> particles_;

  static constexpr PetscInt num_periods_being_kept = 2;
};

#endif  // SRC_DIAGNOSTICS_SIMULATION_BACKUP_H
