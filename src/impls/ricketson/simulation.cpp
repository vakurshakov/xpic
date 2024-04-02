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

  PetscCheck(procs[X] * procs[Y] * procs[Z] == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Ricketson algorithm currently supports only one process");
  ///

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(bounds_), DMDA_STENCIL_BOX, REP3_A(Geom_n), REP3_A(procs), dof, s, REP3(nullptr), &da_));
  PetscCall(DMSetUp(da_));

  PetscCall(DMCreateGlobalVector(da_, &E_));
  PetscCall(DMCreateGlobalVector(da_, &B_));
  PetscCall(setup_norm_gradient());

#if THERE_ARE_PARTICLES
  Particles_parameters parameters = {
    .Np = 1,
    .n  = +1.0,
    .q  = -1.0,
    .m  = +1.0,
    .sort_name = "electrons"
  };
  auto& sort = particles_.emplace_back(parameters);
  sort.add_particle(Point{{geom_x / 2, geom_y / 2, geom_z / 4}, {0.0, 0.0, 0.5}});
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::calculate_B_norm_gradient() {
  PetscFunctionBeginUser;

  Vector3R ***B;
  PetscCall(DMDAVecGetArrayRead(da_, B_, &B));

  PetscReal *B_norm;
  PetscCall(VecGetArrayWrite(B_norm_, &B_norm));

  PetscInt start[3], size[3];
  PetscCall(DMDAGetCorners(da_, REP3_A(&start), REP3_A(&size)));

  for (PetscInt z = start[Z]; z < start[Z] + size[Z]; ++z) {
  for (PetscInt y = start[Y]; y < start[Y] + size[Y]; ++y) {
  for (PetscInt x = start[X]; x < start[X] + size[X]; ++x) {
    B_norm[index(x, y, z)] = B[z][y][x].length();
  }}}

  PetscCall(DMDAVecRestoreArrayRead(da_, B_, &B));
  PetscCall(VecRestoreArrayWrite(B_norm_, &B_norm));

  PetscCall(MatMult(norm_gradient_, B_norm_, B_grad_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::setup_norm_gradient() {
  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(da_, &B_grad_));

  PetscInt start[3], size[3];
  PetscCall(DMDAGetCorners(da_, REP3_A(&start), REP3_A(&size)));

  VecType vtype;
  PetscCall(DMGetVecType(da_, &vtype));

  PetscInt ls = size[X] * size[Y] * size[Z];
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da_), &B_norm_));
  PetscCall(VecSetSizes(B_norm_, ls, PETSC_DETERMINE));
  PetscCall(VecSetType(B_norm_, vtype));
  PetscCall(VecSetUp(B_norm_));

  MatType mtype;
  PetscCall(DMGetMatType(da_, &mtype));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)da_), &norm_gradient_));
  PetscCall(MatSetSizes(norm_gradient_, ls * Vector3R::dim, ls, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(norm_gradient_, mtype));
  PetscCall(MatSetUp(norm_gradient_));

  // The following is the setup of \vec{∇-} operator for |\vec{B}|
  auto remap_with_boundaries = [&](PetscInt& x, PetscInt& y, PetscInt& z) {
    bool success = false;
    if (bounds_[X] == DM_BOUNDARY_PERIODIC && x < 0) { x += geom_nx; success = true; }
    if (bounds_[Y] == DM_BOUNDARY_PERIODIC && y < 0) { y += geom_ny; success = true; }
    if (bounds_[Z] == DM_BOUNDARY_PERIODIC && z < 0) { z += geom_nz; success = true; }
    return success;
  };

  std::vector<Triplet> triplets;
  for (PetscInt z = start[Z]; z < start[Z] + size[Z]; ++z) {
  for (PetscInt y = start[Y]; y < start[Y] + size[Y]; ++y) {
  for (PetscInt x = start[X]; x < start[X] + size[X]; ++x) {
    PetscInt xm = (geom_nx > 1) ? (x - 1) : x;
    PetscInt ym = (geom_ny > 1) ? (y - 1) : y;
    PetscInt zm = (geom_nz > 1) ? (z - 1) : z;

    if (bool success = remap_with_boundaries(xm, ym, zm); !success)
      continue;

    PetscInt cur = index(x, y, z, X);
    triplets.emplace_back(cur, index(x,  y,  z), +1.0 / dx);
    triplets.emplace_back(cur, index(xm, y,  z), -1.0 / dx);

    cur = index(x, y, z, Y);
    triplets.emplace_back(cur, index(x,  y,  z), +1.0 / dy);
    triplets.emplace_back(cur, index(x,  ym, z), -1.0 / dy);

    cur = index(x, y, z, Z);
    triplets.emplace_back(cur, index(x,  y,  z ), +1.0 / dz);
    triplets.emplace_back(cur, index(x,  y,  zm), -1.0 / dz);
  }}}

  for (const auto& [row, col, value] : triplets) {
    // We use `ADD_VALUES` to cancel out values in case of Nx = 1 (or Ny, Nz)
    PetscCall(MatSetValues(norm_gradient_, 1, &row, 1, &col, &value, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(norm_gradient_, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(norm_gradient_, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscInt Simulation::index(PetscInt x, PetscInt y, PetscInt z) {
  return (z * geom_ny + y) * geom_nx + x;
}

PetscInt Simulation::index(PetscInt x, PetscInt y, PetscInt z, PetscInt c) {
  return index(x, y, z) * Vector3I::dim + c;
}


PetscErrorCode Simulation::timestep_implementation(timestep_t timestep) {
  PetscFunctionBeginUser;

  PetscCall(calculate_B_norm_gradient());

  for (auto& sort : particles_) {
    PetscCall(sort.push());
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


Simulation::~Simulation() {
  PetscFunctionBeginUser;
  PetscCallVoid(DMDestroy(&da_));
  PetscCallVoid(VecDestroy(&E_));
  PetscCallVoid(VecDestroy(&B_));
  PetscCallVoid(VecDestroy(&B_grad_));
  PetscCallVoid(VecDestroy(&B_norm_));  // dof = 1
  PetscCallVoid(MatDestroy(&norm_gradient_));
  PetscFunctionReturnVoid();
}

}
