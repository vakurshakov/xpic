#include "simulation.h"

#include <petscdmda.h>

#include "src/vectors/vector_classes.h"

namespace basic {

PetscErrorCode Simulation::initialize_implementation() {
  PetscFunctionBeginUser;
  const Configuration& config = CONFIG();
  Nx = config.size_nx;
  Ny = config.size_ny;
  Nz = config.size_nz;
  dx = config.dx;
  dy = config.dy;
  dz = config.dz;

  const PetscInt dof = Vector3d::dim;
  const PetscInt s = 1; // stencil width (should depend on particle size)

  // We can specify in our config DMBoundaryType and procs number and map it to Create3d
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, R3C(DM_BOUNDARY_NONE), DMDA_STENCIL_BOX, R3CX(N), R3C(PETSC_DECIDE), dof, s, R3C(nullptr), &da_));
  PetscCall(DMSetUp(da_));

  PetscCall(DMCreateGlobalVector(da_, &E_));
  PetscCall(DMCreateGlobalVector(da_, &B_));
  PetscCall(DMCreateMatrix(da_, &rot_p));
  PetscCall(DMCreateMatrix(da_, &rot_m));

  PetscCall(setup_positive_rotor());
  PetscCall(setup_negative_rotor());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::setup_positive_rotor() {
  PetscFunctionBeginUser;

  Vector3<PetscInt> start, end;
  PetscCall(DMDAGetCorners(da_, R3DX(&start), R3DX(&end)));
  end += start;  // Petsc returns size, not end point

  std::vector<Triplet> triplets;
  for (PetscInt z = start.z; z < end.z; ++z) {
  for (PetscInt y = start.y; y < end.y; ++y) {
  for (PetscInt x = start.x; x < end.x; ++x) {
    PetscInt xp = (Nx > 1) ? (x + 1) : x;
    PetscInt yp = (Ny > 1) ? (y + 1) : y;
    PetscInt zp = (Nz > 1) ? (z + 1) : z;

    if (xp >= Nx) continue;
    if (yp >= Ny) continue;
    if (zp >= Nz) continue;

    PetscInt cur = index(x, y, z, X);
    triplets.emplace_back(cur, index(x,  yp, z,  Z), +1.0 / dy);
    triplets.emplace_back(cur, index(x,  y,  z,  Z), -1.0 / dy);
    triplets.emplace_back(cur, index(x,  y,  zp, Y), -1.0 / dz);
    triplets.emplace_back(cur, index(x,  y,  z,  Y), +1.0 / dz);

    cur = index(x, y, z, Y);
    triplets.emplace_back(cur, index(x,  y,  zp, X), +1.0 / dz);
    triplets.emplace_back(cur, index(x,  y,  z,  X), -1.0 / dz);
    triplets.emplace_back(cur, index(xp, y,  z,  Z), -1.0 / dx);
    triplets.emplace_back(cur, index(x,  y,  z,  Z), +1.0 / dx);

    cur = index(x, y, z, Z);
    triplets.emplace_back(cur, index(xp, y,  z,  Y), +1.0 / dx);
    triplets.emplace_back(cur, index(x,  y,  z,  Y), -1.0 / dx);
    triplets.emplace_back(cur, index(x,  yp, z,  X), -1.0 / dy);
    triplets.emplace_back(cur, index(x,  y,  z,  X), +1.0 / dy);
  }}}

  for (const auto& [row, col, value] : triplets) {
    // We use `ADD_VALUES` to cancel out values in case of Nx = 1 (or Ny, Nz)
    PetscCall(MatSetValues(rot_p, 1, &row, 1, &col, &value, ADD_VALUES));
  }

  PetscCall(MatAssemblyBegin(rot_p, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(rot_p, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::setup_negative_rotor() {
  PetscFunctionReturn(PETSC_SUCCESS);
}

constexpr PetscInt Simulation::index(PetscInt x, PetscInt y, PetscInt z, PetscInt c) {
  return ((Ny * z + y) * Nx + x) * Vector3d::dim + c;
}


PetscErrorCode Simulation::timestep_implementation(timestep_t timestep) {
  PetscFunctionReturn(PETSC_SUCCESS);
}

Simulation::~Simulation() {
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&E_));
  PetscCallVoid(VecDestroy(&B_));
  PetscCallVoid(MatDestroy(&rot_p));
  PetscCallVoid(MatDestroy(&rot_m));
  PetscFunctionReturnVoid();
}

}
