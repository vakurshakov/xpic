#include "simulation.h"

#include "src/diagnostics/builders/diagnostic_builder.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"

namespace basic {

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  DM da = world_.da;
  PetscCall(DMCreateGlobalVector(da, &E_));
  PetscCall(DMCreateGlobalVector(da, &B_));
  PetscCall(DMCreateGlobalVector(da, &J_));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rot_dt_p));
  PetscCall(rotor.create_negative(&rot_dt_m));
  PetscCall(MatScale(rot_dt_p, -dt));
  PetscCall(MatScale(rot_dt_m, +dt));

  /// @todo Particles parametrization is needed!
  SortParameters parameters = {
    .sort_name = "electrons",
    .Np = 1,
    .n = +1.0,
    .q = -1.0,
    .m = +1.0,
  };
  auto& sort = particles_.emplace_back(*this, parameters);
  sort.add_particle(Point{{geom_x / 2, geom_y / 2, geom_z / 4}, {0.0, 0.0, 0.9}});

  /// @todo Create a particles_energy diagnostics!
  /// @todo Create a charge_conservation diagnostic!
  PetscCall(build_diagnostics(*this, diagnostics_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::timestep_implementation(timestep_t /* timestep */)
{
  PetscFunctionBeginUser;

  PetscCall(VecSet(J_, 0.0));

  for (auto& sort : particles_) {
    PetscCall(sort.push());
    PetscCall(sort.communicate());
  }

  PetscCall(MatMultAdd(rot_dt_p, E_, B_, B_));  // B (n+1) = B(n) - rot(E) * dt
  PetscCall(MatMultAdd(rot_dt_m, B_, E_, E_));  // E'(n+1) = E(n) + rot(B) * dt
  PetscCall(VecAXPY(E_, -1, J_));              // E (n+1) = E'(n+1) - J

  PetscFunctionReturn(PETSC_SUCCESS);
}

Vec Simulation::get_named_vector(std::string_view name)
{
  if (name == "E")
    return E_;
  if (name == "B")
    return B_;
  if (name == "J")
    return J_;
  throw std::runtime_error("Unknown vector name " + std::string(name));
}

Particles& Simulation::get_named_particles(std::string_view name)
{
  return interfaces::Simulation::get_named_particles(name, particles_);
}


Simulation::~Simulation()
{
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&E_));
  PetscCallVoid(VecDestroy(&B_));
  PetscCallVoid(VecDestroy(&J_));
  PetscCallVoid(MatDestroy(&rot_dt_p));
  PetscCallVoid(MatDestroy(&rot_dt_m));
  PetscFunctionReturnVoid();
}

}  // namespace basic
