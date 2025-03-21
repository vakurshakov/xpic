#ifndef SRC_DIAGNOSTICS_ENERGY_CONSERVATION_H
#define SRC_DIAGNOSTICS_ENERGY_CONSERVATION_H

#include "src/interfaces/diagnostic.h"
#include "src/interfaces/simulation.h"
#include "src/diagnostics/fields_energy.h"
#include "src/diagnostics/particles_energy.h"
#include "src/utils/sync_file.h"

class EnergyConservation : public interfaces::Diagnostic {
public:
  DEFAULT_MOVABLE(EnergyConservation);

  EnergyConservation( //
    const interfaces::Simulation& simulation,
    std::shared_ptr<FieldsEnergy> fields_energy,
    std::shared_ptr<ParticlesEnergy> particles_energy);

  PetscErrorCode diagnose(PetscInt t) final;

protected:
  EnergyConservation(const interfaces::Simulation&);

  virtual PetscErrorCode add_titles();
  virtual PetscErrorCode add_args();

  template<typename T>
  PetscErrorCode write_formatted(
    std::format_string<const T&> fmt, std::vector<T>& container)
  {
    PetscFunctionBeginUser;
    for (const auto& value : container) {
      file_() << std::format(fmt, value);
    }
    file_() << "\n";
    container.clear();
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  void add_arg(PetscReal arg, PetscInt pos = -1);
  void add_title(std::string title, PetscInt pos = -1);

  template<typename T>
  void add(T value, std::vector<T>& container, PetscInt pos = -1)
  {
    if (pos >= 0) {
      auto it = container.begin();
      std::advance(it, pos);
      container.insert(it, value);
    }
    else {
      container.push_back(value);
    }
  }

  const std::string filename_ = //
    "temporal/energy_conservation.txt";

  SyncFile file_;
  std::vector<PetscReal> args_;
  std::vector<std::string> titles_;

  // Vec B0;
  PetscReal dF = 0.0;
  PetscReal dK = 0.0;

  const interfaces::Simulation& simulation;
  std::shared_ptr<FieldsEnergy> fields_energy;
  std::shared_ptr<ParticlesEnergy> particles_energy;
};

#endif  // SRC_DIAGNOSTICS_ENERGY_CONSERVATION_H
