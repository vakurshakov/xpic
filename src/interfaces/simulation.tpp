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
    SortParameters params;

    info.at("sort_name").get_to(params.sort_name);
    PetscCheck(!params.sort_name.empty(), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "SortParameters requires a non-empty name");

    info.at("Np").get_to(params.Np);
    PetscCheck(params.Np > 0 && isfinite(params.Np), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "SortParameters requires finite, positive number of particles per-cel. Given Np = %d", params.Np);

    info.at("n").get_to(params.n);
    PetscCheck(params.n > 0.0 && isfinite(params.n), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "SortParameters requires finite, positive particles density. Given n = %f", params.n);

    info.at("q").get_to(params.q);
    PetscCheck(isfinite(params.q), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "SortParameters requires finite particles charge. Given q = %f", params.q);

    info.at("m").get_to(params.m);
    PetscCheck(params.m > 0.0 && isfinite(params.m), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "SortParameters requires finite, positive particles charge. Given m = %f", params.m);

    bool set_Tall = info.contains("T");

    bool set_Txyz =           //
      info.contains("Tx") ||  //
      info.contains("Ty") ||  //
      info.contains("Tz");

    PetscCheck((!set_Tall && !set_Txyz) || (set_Tall ^ set_Txyz), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "SortParameters requires either no temperature settings, a uniform, or a separate for each individual component temperature specification");

    if (set_Tall) {
      PetscReal T;
      info.at("T").get_to(T);
      params.Tx = T;
      params.Ty = T;
      params.Tz = T;
    }
    else if (set_Txyz) {
      params.Tx = info.value("Tx", (PetscReal)0);
      params.Ty = info.value("Ty", (PetscReal)0);
      params.Tz = info.value("Tz", (PetscReal)0);
    }

    Vector3R T{
      params.Tx,
      params.Ty,
      params.Tz,
    };

    for (Axis a : {X, Y, Z}) {
      PetscCheck(T[a] >= 0.0 && isfinite(T[a]), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
        "SortParameters requires finite and non-negative temperatures. Given T[%d] = %f", (PetscInt)a, T[a]);
    }

    if (info.contains("px"))
      info.at("px").get_to(params.px);
    if (info.contains("py"))
      info.at("py").get_to(params.py);
    if (info.contains("pz"))
      info.at("pz").get_to(params.pz);

    auto&& sort = std::make_shared<PartSpec>(that, params);
    container.emplace_back(sort);
    particles_.emplace_back(sort);

    PetscReal frac = (params.m * mec2);

    Vector3R V{
      sqrt(T[X] / frac),
      sqrt(T[Y] / frac),
      sqrt(T[Z] / frac),
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

    LOG("  {} are added:", params.sort_name);
    LOG("    temperature,      Tx = {:.3e}, Ty = {:.3e}, Tz = {:.3e} [KeV]", REP3_A(T));
    LOG("    thermal velocity, Vx = {:.3e}, Vy = {:.3e}, Vz = {:.3e} [c]", REP3_A(V));
    LOG("    cell-heating, λ_d / Δx = {:.3e} [unit]", CH.abs_max());
    LOG("    cell-traverse, v_th * Δt / Δx = {:.3e} [unit]", CT.abs_max());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace interfaces
