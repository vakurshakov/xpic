#include "simulation.h"

#include "src/utils/utils.h"


namespace ricketson {

PetscErrorCode Simulation::initialize_implementation() {
  PetscFunctionBeginUser;

  PetscMPIInt size;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Ricketson scheme is uniprocessor currently");

  PetscCall(DMCreateGlobalVector(world_.da, &E_));
  PetscCall(DMCreateGlobalVector(world_.da, &B_));
  PetscCall(setup_norm_gradient());

  /// @todo Move this into class Set_approximate_magnetic_mirror.
  const PetscReal current = 3e4;
  const PetscReal radius = 3;
  const PetscReal distance = 15;

  PetscInt rstart[3], rsize[3];
  PetscCall(DMDAGetCorners(world_.da, REP3_A(&rstart), REP3_A(&rsize)));

  Vector3R ***B;
  PetscCall(DMDAVecGetArrayWrite(world_.da, B_, &B));
  for (PetscInt z = rstart[Z]; z < rstart[Z] + rsize[Z]; ++z) {
  for (PetscInt y = rstart[Y]; y < rstart[Y] + rsize[Y]; ++y) {
  for (PetscInt x = rstart[X]; x < rstart[X] + rsize[X]; ++x) {
    PetscReal B0 = 0.5 * current * POW2(radius) * (
      1 / pow(POW2(radius) + POW2((z * dz - 0.5 * geom_z) + 0.5 * distance), 1.5) +
      1 / pow(POW2(radius) + POW2((z * dz - 0.5 * geom_z) - 0.5 * distance), 1.5));

    PetscReal B1 =
      ((z * dz - 0.5 * geom_z) + 0.5 * distance) / (POW2(radius) + POW2((z * dz - 0.5 * geom_z) + 0.5 * distance)) +
      ((z * dz - 0.5 * geom_z) - 0.5 * distance) / (POW2(radius) + POW2((z * dz - 0.5 * geom_z) - 0.5 * distance));

    B[z][y][x].x() = B0 * 1.5 * (x * dx - 0.5 * geom_x) * B1;
    B[z][y][x].y() = B0 * 1.5 * (y * dy - 0.5 * geom_y) * B1;
    B[z][y][x].z() = B0 * 1.0;
  }}}
  PetscCall(DMDAVecRestoreArrayWrite(world_.da, B_, &B));

#if THERE_ARE_PARTICLES
  Sort_parameters parameters = {
    .Np = 1,
    .n  = +1.0,
    .q  = +1.0,
    .m  = +1.0,
    .sort_name = "positron"
  };
  auto& sort = particles_.emplace_back(*this, parameters);

  const PetscReal v_crit = sqrt(9.8342 - 1);
  const PetscReal factor = 0.3;

  Vector3R r = {0, 0.02, 0};
  Vector3R v = {1, 0, factor * v_crit};

  sort.add_particle(Point{r, v});
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::calculate_B_norm_gradient() {
  PetscFunctionBeginUser;

  Vector3R ***B;
  PetscCall(DMDAVecGetArrayRead(world_.da, B_, &B));

  PetscReal *B_norm;
  PetscCall(VecGetArrayWrite(B_norm_, &B_norm));

  PetscInt start[3], size[3];
  PetscCall(DMDAGetCorners(world_.da, REP3_A(&start), REP3_A(&size)));

  for (PetscInt z = start[Z]; z < start[Z] + size[Z]; ++z) {
  for (PetscInt y = start[Y]; y < start[Y] + size[Y]; ++y) {
  for (PetscInt x = start[X]; x < start[X] + size[X]; ++x) {
    B_norm[index(x, y, z)] = B[z][y][x].length();
  }}}

  PetscCall(DMDAVecRestoreArrayRead(world_.da, B_, &B));
  PetscCall(VecRestoreArrayWrite(B_norm_, &B_norm));

  PetscCall(MatMult(norm_gradient_, B_norm_, DB_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::setup_norm_gradient() {
  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(world_.da, &DB_));

  PetscInt start[3], size[3];
  PetscCall(DMDAGetCorners(world_.da, REP3_A(&start), REP3_A(&size)));

  VecType vtype;
  PetscCall(DMGetVecType(world_.da, &vtype));

  PetscInt ls = size[X] * size[Y] * size[Z];
  PetscCall(VecCreate(PetscObjectComm((PetscObject)world_.da), &B_norm_));
  PetscCall(VecSetSizes(B_norm_, ls, PETSC_DETERMINE));
  PetscCall(VecSetType(B_norm_, vtype));
  PetscCall(VecSetUp(B_norm_));

  MatType mtype;
  PetscCall(DMGetMatType(world_.da, &mtype));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)world_.da), &norm_gradient_));
  PetscCall(MatSetSizes(norm_gradient_, ls * Vector3R::dim, ls, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(norm_gradient_, mtype));
  PetscCall(MatSetUp(norm_gradient_));

  /// @todo The code below inherits the problem of proc_y separation, proper indexing should be used.
  /// @note The following is the setup of \vec{∇-} operator for |\vec{B}|. By defining
  /// negative derivative on Yee stencil, we set |\vec{B}| in (i+0.5, j+0.5, k+0.5).
  auto remap_with_boundaries = [&](PetscInt& x, PetscInt& y, PetscInt& z) {
    bool success = false;
    if (world_.bounds[X] == DM_BOUNDARY_PERIODIC && x < 0) { x += geom_nx; success = true; }
    if (world_.bounds[Y] == DM_BOUNDARY_PERIODIC && y < 0) { y += geom_ny; success = true; }
    if (world_.bounds[Z] == DM_BOUNDARY_PERIODIC && z < 0) { z += geom_nz; success = true; }
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
  PetscCallVoid(VecDestroy(&E_));
  PetscCallVoid(VecDestroy(&B_));
  PetscCallVoid(VecDestroy(&DB_));
  PetscCallVoid(VecDestroy(&B_norm_));  // dof = 1
  PetscCallVoid(MatDestroy(&norm_gradient_));
  PetscFunctionReturnVoid();
}

}
