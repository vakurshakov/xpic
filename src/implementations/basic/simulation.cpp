#include "simulation.h"

#include "src/utils/utils.h"
#include "src/vectors/vector_classes.h"

namespace basic {

PetscErrorCode Simulation::initialize_implementation() {
  PetscFunctionBeginUser;

  const PetscInt dof = Vector3d::dim;
  const PetscInt s = 1; // stencil width (should depend on particle size)

  // We can specify in our config DMBoundaryType and procs number and map it to Create3d
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, R3C(DM_BOUNDARY_NONE), DMDA_STENCIL_BOX, R3CX(geom_n), R3C(PETSC_DECIDE), dof, s, R3C(nullptr), &da_));
  PetscCall(DMSetUp(da_));

  PetscCall(DMCreateGlobalVector(da_, &E_));
  PetscCall(DMCreateGlobalVector(da_, &B_));
  PetscCall(DMCreateMatrix(da_, &rot_dt_p));
  PetscCall(DMCreateMatrix(da_, &rot_dt_m));

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
    PetscInt xp = (geom_nx > 1) ? (x + 1) : x;
    PetscInt yp = (geom_ny > 1) ? (y + 1) : y;
    PetscInt zp = (geom_nz > 1) ? (z + 1) : z;

    // Simplest boundary condition
    if (xp >= geom_nx) continue;
    if (yp >= geom_ny) continue;
    if (zp >= geom_nz) continue;

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
    PetscCall(MatSetValues(rot_dt_p, 1, &row, 1, &col, &value, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(rot_dt_p, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(rot_dt_p, MAT_FINAL_ASSEMBLY));

  PetscCall(MatScale(rot_dt_p, -dt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::setup_negative_rotor() {
  PetscFunctionBeginUser;

  Vector3<PetscInt> start, end;
  PetscCall(DMDAGetCorners(da_, R3DX(&start), R3DX(&end)));
  end += start;  // Petsc returns size, not end point

  std::vector<Triplet> triplets;
  for (PetscInt z = start.z; z < end.z; ++z) {
  for (PetscInt y = start.y; y < end.y; ++y) {
  for (PetscInt x = start.x; x < end.x; ++x) {
    PetscInt xm = (geom_nx > 1) ? (x - 1) : x;
    PetscInt ym = (geom_ny > 1) ? (y - 1) : y;
    PetscInt zm = (geom_nz > 1) ? (z - 1) : z;

    if (xm < 0) continue;
    if (ym < 0) continue;
    if (zm < 0) continue;

    PetscInt cur = index(x, y, z, X);
    triplets.emplace_back(cur, index(x,  y,  z,  Z), +1.0 / dx);
    triplets.emplace_back(cur, index(x,  ym, z,  Z), -1.0 / dx);
    triplets.emplace_back(cur, index(x,  y,  z,  Y), -1.0 / dx);
    triplets.emplace_back(cur, index(x,  y,  zm, Y), +1.0 / dx);

    cur = index(x, y, z, Y);
    triplets.emplace_back(cur, index(x,  y,  z,  X), +1.0 / dx);
    triplets.emplace_back(cur, index(x,  y,  zm, X), -1.0 / dx);
    triplets.emplace_back(cur, index(x,  y,  z,  Z), -1.0 / dx);
    triplets.emplace_back(cur, index(xm, y,  z,  Z), +1.0 / dx);

    cur = index(x, y, z, Z);
    triplets.emplace_back(cur, index(x,  y,  z,  Y), +1.0 / dx);
    triplets.emplace_back(cur, index(xm, y,  z,  Y), -1.0 / dx);
    triplets.emplace_back(cur, index(x,  y,  z,  X), -1.0 / dx);
    triplets.emplace_back(cur, index(x,  ym, z,  X), +1.0 / dx);
  }}}

  for (const auto& [row, col, value] : triplets) {
    // We use `ADD_VALUES` to cancel out values in case of Nx = 1 (or Ny, Nz)
    PetscCall(MatSetValues(rot_dt_m, 1, &row, 1, &col, &value, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(rot_dt_m, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(rot_dt_m, MAT_FINAL_ASSEMBLY));

  PetscCall(MatScale(rot_dt_m, +dt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscInt Simulation::index(PetscInt x, PetscInt y, PetscInt z, PetscInt c) {
  return ((geom_ny * z + y) * geom_nx + x) * Vector3d::dim + c;
}


PetscErrorCode Simulation::timestep_implementation(timestep_t timestep) {
  PetscFunctionBeginUser;

  // Field source
  PetscCall(VecSetValue(B_, index(geom_nx / 2, geom_ny / 2, geom_nz / 2, Z), 1.0, ADD_VALUES));
  PetscCall(VecAssemblyBegin(B_));
  PetscCall(VecAssemblyEnd(B_));

  // Solving Maxwell's equations using FDTD
  PetscCall(MatMultAdd(rot_dt_p, E_, B_, B_));  // rot(E) = - ∂B / ∂t
  PetscCall(MatMultAdd(rot_dt_m, B_, E_, E_));  // rot(B) = + ∂E / ∂t

  /// @todo Write a simplest diagnostics of fields energy

  PetscFunctionReturn(PETSC_SUCCESS);
}

Simulation::~Simulation() {
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&E_));
  PetscCallVoid(VecDestroy(&B_));
  PetscCallVoid(MatDestroy(&rot_dt_p));
  PetscCallVoid(MatDestroy(&rot_dt_m));
  PetscFunctionReturnVoid();
}

}
