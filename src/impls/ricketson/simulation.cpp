#include "simulation.h"

#include "src/utils/utils.h"
#include "src/vectors/vector_classes.h"

namespace ricketson {

PetscErrorCode Simulation::initialize_implementation() {
  PetscFunctionBeginUser;

  /// @todo The following can be moved into the base class, or 3D simulation base class.
  /// @todo Diagnostics are also independent of particle container, so they can be reused.
  const PetscInt dof = Vector3R::dim;
  const PetscInt s = shape_radius;

  const Configuration& config = CONFIG();
  const Configuration::json_t& geometry = config.json.at("Geometry");

  std::string boundary_type_str[3];
  geometry.at("da_boundary_x").get_to(boundary_type_str[X]);
  geometry.at("da_boundary_y").get_to(boundary_type_str[Y]);
  geometry.at("da_boundary_z").get_to(boundary_type_str[Z]);

  auto to_boundary_type = [](const std::string& str) {
    if (str == "DM_BOUNDARY_PERIODIC") return DM_BOUNDARY_PERIODIC;
    if (str == "DM_BOUNDARY_GHOSTED") return DM_BOUNDARY_GHOSTED;
    return DM_BOUNDARY_NONE;
  };

  bounds_[X] = to_boundary_type(boundary_type_str[X]);
  bounds_[Y] = to_boundary_type(boundary_type_str[Y]);
  bounds_[Z] = to_boundary_type(boundary_type_str[Z]);

  PetscInt procs[3];
  geometry.at("da_processors_x").get_to(procs[X]);
  geometry.at("da_processors_y").get_to(procs[Y]);
  geometry.at("da_processors_z").get_to(procs[Z]);
  ///

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(bounds_), DMDA_STENCIL_BOX, REP3_A(Geom_n), REP3_A(procs), dof, s, REP3(nullptr), &da_));
  PetscCall(DMSetUp(da_));

#if THERE_ARE_PARTICLES
  Particles_parameters parameters = {
    .Np = 1,
    .n  = +1.0,
    .q  = -1.0,
    .m  = +1.0,
    .sort_name = "electrons"
  };
  auto& sort = particles_.emplace_back(parameters);
  sort.add_particle(Point{{geom_x / 2, geom_y / 2, geom_z / 4}, {0.0, 0.0, 0.9}});
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::timestep_implementation(timestep_t timestep) {
  PetscFunctionBeginUser;

  for (auto& sort : particles_) {
    PetscCall(sort.push());
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


Simulation::~Simulation() {
  PetscFunctionBeginUser;
  PetscCallVoid(DMDestroy(&da_));
  PetscFunctionReturnVoid();
}

}
