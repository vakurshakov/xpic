#include "simulation.h"

#include "src/utils/utils.h"
#include "src/vectors/vector3.h"
#include "src/vectors/rotor.h"
#include "src/impls/basic/diagnostics/builders/diagnostic_builder.h"

namespace basic {

PetscErrorCode Simulation::initialize_implementation() {
  PetscFunctionBeginUser;

  const PetscInt dof = Vector3R::dim;
  const PetscInt s = shape_radius;

  PetscInt procs[3];
  Configuration::get_boundaries_type(REP3_A(bounds_));
  Configuration::get_processors(REP3_A(procs));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(bounds_), DMDA_STENCIL_BOX, REP3_A(Geom_n), REP3_A(procs), dof, s, REP3(nullptr), &da_));
  PetscCall(DMSetUp(da_));

  PetscCall(DMCreateGlobalVector(da_, &E_));
  PetscCall(DMCreateGlobalVector(da_, &B_));
  PetscCall(DMCreateGlobalVector(da_, &J_));
  PetscCall(DMCreateMatrix(da_, &rot_dt_p));
  PetscCall(DMCreateMatrix(da_, &rot_dt_m));

  Rotor rotor(da_);
  PetscCall(rotor.set_positive(rot_dt_p));
  PetscCall(rotor.set_negative(rot_dt_m));
  PetscCall(MatScale(rot_dt_p, -dt));
  PetscCall(MatScale(rot_dt_m, +dt));

#if THERE_ARE_PARTICLES
  /// @todo Particles parametrisation is needed!
  Particles_parameters parameters = {
    .Np = 1,
    .n  = +1.0,
    .q  = -1.0,
    .m  = +1.0,
    .sort_name = "electrons"
  };
  auto& sort = particles_.emplace_back(*this, parameters);
  sort.add_particle(Point{{geom_x / 2, geom_y / 2, geom_z / 4}, {0.0, 0.0, 0.9}});
#endif

  /// @todo Create a particles_energy diagnostics!
  /// @todo Create a charge_conservation diagnostic!
  PetscCall(build_diagnostics(*this, diagnostics_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::timestep_implementation(timestep_t timestep) {
  PetscFunctionBeginUser;

  PetscCall(VecSet(J_, 0.0));

  for (auto& sort : particles_) {
    PetscCall(sort.push());
    PetscCall(sort.communicate());
  }

  PetscCall(MatMultAdd(rot_dt_p, E_, B_, B_));  // B (n+1) = B(n) - rot(E) * dt
  PetscCall(MatMultAdd(rot_dt_m, B_, E_, E_));  // E'(n+1) = E(n) + rot(B) * dt
  PetscCall(VecAXPY(E_, -1, J_));               // E (n+1) = E'(n+1) - J

  PetscFunctionReturn(PETSC_SUCCESS);
}


Simulation::~Simulation() {
  PetscFunctionBeginUser;
  PetscCallVoid(DMDestroy(&da_));
  PetscCallVoid(VecDestroy(&E_));
  PetscCallVoid(VecDestroy(&B_));
  PetscCallVoid(VecDestroy(&J_));
  PetscCallVoid(MatDestroy(&rot_dt_p));
  PetscCallVoid(MatDestroy(&rot_dt_m));
  PetscFunctionReturnVoid();
}

}
