#include "src/diagnostics/charge_conservation.h"
#include "src/impls/hybrid_drift_kinetic/diagnostic.h"
#include "src/impls/hybrid_drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

class InspectSimulation : public hybrid_drift_kinetic::Simulation {
public:
  using interfaces::Simulation::diagnostics_;
};

static void overwrite_config(const std::string& name,
  const std::vector<std::string>& models, bool include_particles);
static PetscErrorCode check_case(const std::string& name,
  const std::vector<std::string>& models, bool include_particles = true);
static PetscErrorCode check_invalid_model();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
  PetscCall(check_case("mixed",
    {"kinetic", "drift_kinetic", "kinetic", "", "drift_kinetic"}));
  PetscCall(check_case("dk", {"drift_kinetic", ""}));
  PetscCall(check_case("kinetic", {"kinetic", "kinetic"}));
  PetscCall(check_case("empty", {}));
  PetscCall(check_case("omitted", {}, false));
  PetscCall(check_invalid_model());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Sorts are named "sort<i>"; an empty model omits the "model" key.
static void overwrite_config(const std::string& name,
  const std::vector<std::string>& models, bool include_particles)
{
  dx = dy = dz = 1;
  geom_nx = geom_ny = geom_nz = 8;
  geom_x = geom_y = geom_z = 8;
  dt = 1;
  geom_nt = 0;
  geom_t = 0;

  Configuration::array_t sorts;
  for (std::size_t i = 0; i < models.size(); i++) {
    Configuration::json_t sort{
      {"sort_name", "sort" + std::to_string(i)},
      {"Np", 2},
      {"n", 3},
      {"q", 1},
      {"m", 2},
      {"push", {{"atol", 1e-14}, {"rtol", 1e-14}}},
    };
    if (!models[i].empty())
      sort["model"] = models[i];
    sorts.push_back(sort);
  }

  Configuration::json_t config{
    {"Simulation", "hybrid_drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__) / name},
    {"DriftKineticPush",
      {{"coord_is_gc", true}, {"substepping", true}, {"max_substep_depth", 2},
        {"fail_on_any_push_nonconvergence", true}}},
    {"Geometry",
      {
        {"x", geom_x},
        {"y", geom_y},
        {"z", geom_z},
        {"t", geom_t},
        {"dx", dx},
        {"dy", dy},
        {"dz", dz},
        {"dt", dt},
        {"diagnose_period", dt},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      }},
    {"Fields", {{"evolve", false}}},
    {"Presets",
      {
        {{"command", "SetField"}, {"field", "B"},
          {"setter", {{"name", "SetUniformField"}, {"value", {0, 0, 2}}}}},
      }},
  };
  if (include_particles)
    config["Particles"] = sorts;
  Configuration::overwrite(std::move(config));
}

// Checks the sort types, the order of the containers and the diagnostics.
static PetscErrorCode check_case(const std::string& name,
  const std::vector<std::string>& models, bool include_particles)
{
  PetscFunctionBeginUser;
  overwrite_config(name, models, include_particles);
  InspectSimulation simulation;
  PetscCall(simulation.initialize());
  const auto& all = simulation.interfaces::Simulation::particles_;

  PetscCheck(all.size() == models.size(), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "The shared container must have one entry per configured sort");

  std::size_t dk_count = 0, kinetic_count = 0;
  for (std::size_t i = 0; i < all.size(); i++) {
    const auto& sort = all[i];
    const bool kinetic = models[i] == "kinetic";

    PetscCheck(sort->parameters.sort_name == "sort" + std::to_string(i),
      PETSC_COMM_WORLD, PETSC_ERR_USER,
      "The shared container must preserve configuration order");

    if (kinetic) {
      PetscCheck(dynamic_cast<hybrid_drift_kinetic::Particles*>(sort.get()) &&
          kinetic_count < simulation.kinetic_particles_.size() &&
          simulation.kinetic_particles_[kinetic_count++].get() == sort.get(),
        PETSC_COMM_WORLD, PETSC_ERR_USER,
        "The kinetic container must refer to the same objects in configuration order");
    }
    else {
      auto* dk = dynamic_cast<drift_kinetic::Particles*>(sort.get());
      PetscCheck(dk && dk_count < simulation.particles_.size() &&
          simulation.particles_[dk_count++].get() == dk &&
          dk->get_fail_on_any_push_nonconvergence(),
        PETSC_COMM_WORLD, PETSC_ERR_USER,
        "The DK container must share the configured objects and receive DK push settings");
    }

    ParticleStorage::Attributes fields;
    PetscCall(sort->storage.get_attributes(fields));
    const bool has_mu = fields.dof.contains("mu_p");
    PetscCall(sort->storage.restore_attributes(fields));
    PetscCheck(has_mu == !kinetic, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Only DK storage must contain the magnetic moment");
  }

  PetscCheck(dk_count == simulation.particles_.size() &&
      kinetic_count == simulation.kinetic_particles_.size(),
    PETSC_COMM_WORLD, PETSC_ERR_USER, "Typed containers must not contain extra sorts");

  PetscInt energies = 0, convergence = 0, charge = 0;
  for (const auto& diagnostic : simulation.diagnostics_) {
    energies += dynamic_cast<hybrid_drift_kinetic::EnergyConservation*>(
                  diagnostic.get()) != nullptr;
    convergence += dynamic_cast<hybrid_drift_kinetic::ConvergenceHistory*>(
                     diagnostic.get()) != nullptr;
    charge += dynamic_cast<ChargeConservation*>(diagnostic.get()) != nullptr;
  }
  PetscCheck(energies == 1 && convergence == 1 && charge == 1 &&
      simulation.diagnostics_.size() == 3,
    PETSC_COMM_WORLD, PETSC_ERR_USER,
    "The hybrid must register each diagnostic exactly once");

  PetscCall(simulation.finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode check_invalid_model()
{
  PetscFunctionBeginUser;
  overwrite_config("invalid", {"unknown"}, true);
  InspectSimulation simulation;
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, nullptr));
  const PetscErrorCode error = simulation.initialize();
  PetscCall(PetscPopErrorHandler());
  PetscCheck(error == PETSC_ERR_ARG_WRONG, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "An unknown model must be rejected");
  PetscCheck(simulation.interfaces::Simulation::particles_.empty(),
    PETSC_COMM_WORLD, PETSC_ERR_USER,
    "An invalid model must not create a particle sort");
  PetscCall(simulation.finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
