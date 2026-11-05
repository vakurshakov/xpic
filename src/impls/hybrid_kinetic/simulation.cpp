#include "simulation.h"

#include "src/impls/hybrid_kinetic/diagnostic.h"
#include "src/utils/configuration.h"
#include "src/utils/utils.h"

namespace hybrid_kinetic {

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscCall(drift_kinetic::Simulation::initialize_implementation());
  PetscCall(SNESSetFunction(snes, NULL, Simulation::form_iteration, this));
  PetscCall(init_kinetic_particles());
  PetscCall(DMCreateGlobalVector(da_EB, &x_last));

  // The base adds the drift-kinetic `EnergyConservation` last,
  // it misses the kinetic sorts.
  PetscCheck(dynamic_cast<drift_kinetic::EnergyConservation*>(diagnostics_.back().get()), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
    "The last diagnostic of drift_kinetic::Simulation is expected to be EnergyConservation");

  diagnostics_.back() = std::make_unique<EnergyConservation>(*this);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&x_last));
  PetscCall(drift_kinetic::Simulation::finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::timestep_implementation(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  for (auto& sort : particles_)
    PetscCall(sort->prepare_storage());
  for (auto& sort : kinetic_particles_)
    PetscCall(sort->prepare_storage());

  if (!evolve_fields) {
    PetscCall(VecCopy(E, E_hk));
    PetscCall(VecCopy(B, B_hk));
    PetscCall(form_current());
  }
  else {
    PetscCall(to_snes(E, B, sol));

    PetscCall(SNESSolve(snes, NULL, sol));
    PetscCall(SNESGetIterationNumber(snes, &snes_iteration_number));

    LOG("SNESSolve() has finished, SNESConvergedReasonView():");
    PetscCall(SNESConvergedReasonView(snes, PETSC_VIEWER_STDOUT_WORLD));

    SNESConvergedReason reason;
    PetscCall(SNESGetConvergedReason(snes, &reason));

    PetscCheck(reason >= 0, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED,
      "SNESSolve has not converged, reason: %s", SNESConvergedReasons[reason]);

    PetscCall(SNESGetSolution(snes, &sol));
    PetscCall(from_snes(sol, E_hk, B_hk));

    // `SNESNGMRES` may return a point other than the last pushed one,
    // then the particles and sources are made consistent with it.
    PetscBool is_last;
    PetscCall(VecEqual(sol, x_last, &is_last));

    if (!is_last) {
      LOG("The returned solution is not the last pushed point, pushing again");
      PetscCall(form_current());
    }

    PetscCall(VecAXPBY(E, 2, -1, E_hk));
    PetscCall(VecAXPBY(B, 2, -1, B_hk));
  }

  for (auto& sort : particles_)
    PetscCall(sort->update_cells());
  for (auto& sort : kinetic_particles_)
    PetscCall(sort->update_cells());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_iteration(SNES, Vec vx, Vec vf, void* ctx)
{
  PetscFunctionBeginUser;
  auto* simulation = (Simulation*)ctx;
  PetscCall(VecCopy(vx, simulation->x_last));
  PetscCall(simulation->from_snes(vx, simulation->E_hk, simulation->B_hk));
  PetscCall(simulation->form_current());
  PetscCall(simulation->form_function(vf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// All sorts are created by the base as drift-kinetic ones,
// the "kinetic" ones are rebuilt here with the same parameters.
PetscErrorCode Simulation::init_kinetic_particles()
{
  PetscFunctionBeginUser;
  const Configuration::json_t& json = CONFIG().json;
  auto it = json.find("Particles");

  if (it == json.end())
    PetscFunctionReturn(PETSC_SUCCESS);

  auto& all_particles = interfaces::Simulation::particles_;

  for (auto&& info : *it) {
    const std::string model = info.value("model", "drift_kinetic");

    PetscCheck(model == "kinetic" || model == "drift_kinetic", PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "Unknown particles model \"%s\", expected \"kinetic\" or \"drift_kinetic\"", model.c_str());

    if (model != "kinetic")
      continue;

    const std::string name = info.at("sort_name");
    auto is_named = [&](const auto& sort)
    {
      return sort->parameters.sort_name == name;
    };

    auto dk = std::find_if(particles_.begin(), particles_.end(), is_named);
    auto base =
      std::find_if(all_particles.begin(), all_particles.end(), is_named);

    auto sort = std::make_shared<KineticParticles>(world, (*dk)->parameters);
    PetscCall((*dk)->finalize());
    particles_.erase(dk);
    *base = sort;
    kinetic_particles_.emplace_back(sort);

    LOG("  {} are described by the kinetic model", name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_current()
{
  PetscFunctionBeginUser;
  // Guiding centers: zeroes `J`, `M` and moves `E_hk`, `B_hk` to locals.
  PetscCall(drift_kinetic::Simulation::form_current());

  PetscCall(DMDAVecGetArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecGetArrayRead(da, B_loc, &B_arr));

  for (auto& sort : kinetic_particles_) {
    PetscCall(VecSet(sort->J, 0));
    PetscCall(VecSet(sort->J_loc, 0));

    sort->E_arr = E_arr;
    sort->B_arr = B_arr;
    PetscCall(sort->form_iteration());
    PetscCall(VecAXPY(J, 1, sort->J));

    LOG("Push \"{}\": avg it={}", sort->parameters.sort_name, sort->get_average_iteration_number());
  }

  PetscCall(DMDAVecRestoreArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, B_loc, &B_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace hybrid_kinetic
