#include "rotor.h"

#include "src/pch.h"
#include "src/vectors/vector_classes.h"

Rotor::Rotor(DM da) : da_(da) {
  PetscCallVoid(DMDAGetInfo(da_, nullptr, REP3(nullptr), REP3(nullptr), nullptr, nullptr, REP3_A(&bounds_), nullptr));
  PetscCallVoid(DMDAGetCorners(da_, REP3_A(&start_), REP3_A(&size_)));
}

PetscErrorCode Rotor::set_positive(Mat mat) {
  PetscFunctionBeginUser;
  PetscCall(MatSetOption(mat, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
  PetscCall(MatSetOption(mat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));

  std::vector<Triplet> triplets;
  for (PetscInt z = start_[Z]; z < start_[Z] + size_[Z]; ++z) {
  for (PetscInt y = start_[Y]; y < start_[Y] + size_[Y]; ++y) {
  for (PetscInt x = start_[X]; x < start_[X] + size_[X]; ++x) {
    PetscInt xp = (geom_nx > 1) ? (x + 1) : x;
    PetscInt yp = (geom_ny > 1) ? (y + 1) : y;
    PetscInt zp = (geom_nz > 1) ? (z + 1) : z;

    if (bool success = remap_positive(xp, yp, zp); !success)
      continue;

    PetscInt cur = index(x, y, z, X);
    PetscPrintf(MPI_COMM_SELF, "[%d] (%d %d %d %d)=%3d", x, y, z, (int)X, cur);

    triplets.emplace_back(cur, index(x,  yp, z,  Z), +1.0 / dy);
    triplets.emplace_back(cur, index(x,  y,  z,  Z), -1.0 / dy);
    triplets.emplace_back(cur, index(x,  y,  zp, Y), -1.0 / dz);
    triplets.emplace_back(cur, index(x,  y,  z,  Y), +1.0 / dz);

    cur = index(x, y, z, Y);
    PetscPrintf(MPI_COMM_SELF, "[%d] (%d %d %d %d)=%3d", x, y, z, (int)Y, cur);

    triplets.emplace_back(cur, index(x,  y,  zp, X), +1.0 / dz);
    triplets.emplace_back(cur, index(x,  y,  z,  X), -1.0 / dz);
    triplets.emplace_back(cur, index(xp, y,  z,  Z), -1.0 / dx);
    triplets.emplace_back(cur, index(x,  y,  z,  Z), +1.0 / dx);

    cur = index(x, y, z, Z);
    PetscPrintf(MPI_COMM_SELF, "[%d] (%d %d %d %d)=%3d", x, y, z, (int)Z, cur);

    triplets.emplace_back(cur, index(xp, y,  z,  Y), +1.0 / dx);
    triplets.emplace_back(cur, index(x,  y,  z,  Y), -1.0 / dx);
    triplets.emplace_back(cur, index(x,  yp, z,  X), -1.0 / dy);
    triplets.emplace_back(cur, index(x,  y,  z,  X), +1.0 / dy);
  }}}

  for (const auto& [row, col, value] : triplets) {
    // We use `ADD_VALUES` to cancel out values in case of Nx = 1 (or Ny, Nz)
    PetscCall(MatSetValues(mat, 1, &row, 1, &col, &value, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Rotor::set_negative(Mat mat) {
  PetscFunctionBeginUser;
  PetscCall(MatSetOption(mat, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
  PetscCall(MatSetOption(mat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));

  std::vector<Triplet> triplets;
  for (PetscInt z = start_[Z]; z < start_[Z] + size_[Z]; ++z) {
  for (PetscInt y = start_[Y]; y < start_[Y] + size_[Y]; ++y) {
  for (PetscInt x = start_[X]; x < start_[X] + size_[X]; ++x) {
    PetscInt xm = (geom_nx > 1) ? (x - 1) : x;
    PetscInt ym = (geom_ny > 1) ? (y - 1) : y;
    PetscInt zm = (geom_nz > 1) ? (z - 1) : z;

    if (bool success = remap_negative(xm, ym, zm); !success)
      continue;

    PetscInt cur = index(x, y, z, X);
    triplets.emplace_back(cur, index(x,  y,  z,  Z), +1.0 / dy);
    triplets.emplace_back(cur, index(x,  ym, z,  Z), -1.0 / dy);
    triplets.emplace_back(cur, index(x,  y,  z,  Y), -1.0 / dz);
    triplets.emplace_back(cur, index(x,  y,  zm, Y), +1.0 / dz);

    cur = index(x, y, z, Y);
    triplets.emplace_back(cur, index(x,  y,  z,  X), +1.0 / dz);
    triplets.emplace_back(cur, index(x,  y,  zm, X), -1.0 / dz);
    triplets.emplace_back(cur, index(x,  y,  z,  Z), -1.0 / dx);
    triplets.emplace_back(cur, index(xm, y,  z,  Z), +1.0 / dx);

    cur = index(x, y, z, Z);
    triplets.emplace_back(cur, index(x,  y,  z,  Y), +1.0 / dx);
    triplets.emplace_back(cur, index(xm, y,  z,  Y), -1.0 / dx);
    triplets.emplace_back(cur, index(x,  y,  z,  X), -1.0 / dy);
    triplets.emplace_back(cur, index(x,  ym, z,  X), +1.0 / dy);
  }}}

  for (const auto& [row, col, value] : triplets) {
    // We use `ADD_VALUES` to cancel out values in case of Nx = 1 (or Ny, Nz)
    PetscCall(MatSetValues(mat, 1, &row, 1, &col, &value, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscInt Rotor::index(PetscInt x, PetscInt y, PetscInt z, PetscInt c) {
  return ((z * geom_ny + y) * geom_nx + x) * Vector3I::dim + c;
}


/// @todo Check remap for periodic boundaries and geom_ns == 1 (ADD_VALUES)
bool Rotor::remap_positive(PetscInt& x, PetscInt& y, PetscInt& z) {
  bool success = false;
  if (bounds_[X] == DM_BOUNDARY_PERIODIC && x >= geom_nx) { x -= geom_nx; success = true; }
  if (bounds_[Y] == DM_BOUNDARY_PERIODIC && y >= geom_ny) { y -= geom_ny; success = true; }
  if (bounds_[Z] == DM_BOUNDARY_PERIODIC && z >= geom_nz) { z -= geom_nz; success = true; }
  return success;
}

bool Rotor::remap_negative(PetscInt& x, PetscInt& y, PetscInt& z) {
  bool success = false;
  if (bounds_[X] == DM_BOUNDARY_PERIODIC && x < 0) { x += geom_nx; success = true; }
  if (bounds_[Y] == DM_BOUNDARY_PERIODIC && y < 0) { y += geom_ny; success = true; }
  if (bounds_[Z] == DM_BOUNDARY_PERIODIC && z < 0) { z += geom_nz; success = true; }
  return success;
}
