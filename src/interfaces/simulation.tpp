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

    PetscReal T = std::hypot(parameters.Tx, parameters.Ty, parameters.Tz);
    PetscReal V = std::sqrt(T / (parameters.m * 511.0));
    PetscReal H = std::hypot(Dx[X] / V, Dx[Y] / V, Dx[Z] / V);

    LOG("  {} are added:", parameters.sort_name);
    LOG("    temperature,         T = {:.3e} [KeV]", T);
    LOG("    thermal velocity, v_th = {:.3e} [c]", V);
    LOG("    cell-heating, Dx / L_d = {:.3e} [unit]", H);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace interfaces
