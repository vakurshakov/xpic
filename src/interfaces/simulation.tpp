#include "simulation.h"
#include "src/utils/configuration.h"

namespace interfaces {

template<class SimSpec, class PartSpec>
PetscErrorCode Simulation::init_particles(
  SimSpec& that, std::vector<std::shared_ptr<PartSpec>>& container)
{
  PetscFunctionBeginUser;
  LOG("Configuring particles");

  const Configuration::json_t& json = CONFIG().json;
  auto it = json.find("Particles");

  if (it == json.end() || it->empty())
    PetscFunctionReturn(PETSC_SUCCESS);

  for (auto&& info : *it) {
    if (!info.contains("sort_name"))
      continue;

    SortParameters parameters;
    info.at("sort_name").get_to(parameters.sort_name);
    info.at("Np").get_to(parameters.Np);
    info.at("n").get_to(parameters.n);
    info.at("q").get_to(parameters.q);
    info.at("m").get_to(parameters.m);

    if (info.contains("T")) {
      PetscReal T;
      info.at("T").get_to(T);
      parameters.Tx = T;
      parameters.Ty = T;
      parameters.Tz = T;
    }
    else {
      info.at("Tx").get_to(parameters.Tx);
      info.at("Ty").get_to(parameters.Ty);
      info.at("Tz").get_to(parameters.Tz);
    }

    auto&& sort = std::make_shared<PartSpec>(that, parameters);
    container.emplace_back(sort);
    particles_.emplace_back(sort);

    PetscReal frac = (parameters.m * 511.0);

    Vector3R T{
      parameters.Tx,
      parameters.Ty,
      parameters.Tz,
    };

    Vector3R V{
      std::sqrt(T[X] / frac),
      std::sqrt(T[Y] / frac),
      std::sqrt(T[Z] / frac),
    };

    Vector3R CH{
      V[X] / Dx[X],
      V[Y] / Dx[Y],
      V[Z] / Dx[Z],
    };

    Vector3R CT{
      V[X] * dt / Dx[X],
      V[Y] * dt / Dx[Y],
      V[Z] * dt / Dx[Z],
    };

    LOG("  {} are added:", parameters.sort_name);
    LOG("    temperature (avg.),  T = {:.3e} [KeV]", T.length());
    LOG("    thermal velocity, v_th = {:.3e} [c]", V.length());
    LOG("    cell-heating, λ_d / Δx = {:.3e} [unit]", CH.abs_max());
    LOG("    cell-traverse, v_th * Δt / Δx = {:.3e} [unit]", CT.abs_max());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace interfaces
