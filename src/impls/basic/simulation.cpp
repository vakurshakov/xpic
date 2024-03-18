#include "simulation.h"

#include "src/utils/utils.h"
#include "src/vectors/vector_classes.h"
#include "src/impls/basic/diagnostics/builders/diagnostic_builder.h"

namespace basic {

PetscErrorCode Simulation::initialize_implementation() {
  PetscFunctionBeginUser;

  const PetscInt dof = Vector3_dim;
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

  bounds[X] = to_boundary_type(boundary_type_str[X]);
  bounds[Y] = to_boundary_type(boundary_type_str[Y]);
  bounds[Z] = to_boundary_type(boundary_type_str[Z]);

  PetscInt procs[3];
  geometry.at("da_processors_x").get_to(procs[X]);
  geometry.at("da_processors_y").get_to(procs[Y]);
  geometry.at("da_processors_z").get_to(procs[Z]);

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(bounds), DMDA_STENCIL_BOX, REP3_A(Geom_n), REP3_A(procs), dof, s, REP3(nullptr), &da_));
  PetscCall(DMSetUp(da_));

  PetscCall(DMCreateGlobalVector(da_, &E_));
  PetscCall(DMCreateGlobalVector(da_, &B_));
  PetscCall(DMCreateGlobalVector(da_, &J_));
  PetscCall(DMCreateMatrix(da_, &rot_dt_p));
  PetscCall(DMCreateMatrix(da_, &rot_dt_m));

  PetscCall(setup_positive_rotor());
  PetscCall(setup_negative_rotor());

#if THERE_ARE_PARTICLES
  Particles_parameters parameters = {
    .Np = 1,
    .n  = +1.0,
    .q  = -1.0,
    .m  = +1.0,
    .sort_name = "electrons"
  };
  auto& sort = particles_.emplace_back(*this, parameters);
  sort.add_particle(Point{{geom_x / 2, geom_y / 4, geom_z / 2}, {0.0, 1.0, 0.0}});
#endif

  PetscCall(build_diagnostics(*this, diagnostics_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::setup_positive_rotor() {
  PetscFunctionBeginUser;

  Vector3<PetscInt> start, end;
  PetscCall(DMDAGetCorners(da_, REP3_A(&start), REP3_A(&end)));
  end += start;  // Petsc returns size, not end point

  /// @todo Check remap for periodic boundaries and geom_ns == 1 (ADD_VALUES)
  auto remap_with_boundaries = [&](PetscInt& x, PetscInt& y, PetscInt& z) {
    bool success = false;
    if (bounds[X] == DM_BOUNDARY_PERIODIC && x >= geom_nx) { x -= geom_nx; success = true; }
    if (bounds[Y] == DM_BOUNDARY_PERIODIC && y >= geom_ny) { y -= geom_ny; success = true; }
    if (bounds[Z] == DM_BOUNDARY_PERIODIC && z >= geom_nz) { z -= geom_nz; success = true; }
    return success;
  };

  std::vector<Triplet> triplets;
  for (PetscInt z = start.z(); z < end.z(); ++z) {
  for (PetscInt y = start.y(); y < end.y(); ++y) {
  for (PetscInt x = start.x(); x < end.x(); ++x) {
    PetscInt xp = (geom_nx > 1) ? (x + 1) : x;
    PetscInt yp = (geom_ny > 1) ? (y + 1) : y;
    PetscInt zp = (geom_nz > 1) ? (z + 1) : z;

    if (bool success = remap_with_boundaries(xp, yp, zp); !success)
      continue;

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
  PetscCall(DMDAGetCorners(da_, REP3_A(&start), REP3_A(&end)));
  end += start;  // Petsc returns size, not end point

  auto remap_with_boundaries = [&](PetscInt& x, PetscInt& y, PetscInt& z) {
    bool success = false;
    if (bounds[X] == DM_BOUNDARY_PERIODIC && x < 0) { x += geom_nx; success = true; }
    if (bounds[Y] == DM_BOUNDARY_PERIODIC && y < 0) { y += geom_ny; success = true; }
    if (bounds[Z] == DM_BOUNDARY_PERIODIC && z < 0) { z += geom_nz; success = true; }
    return success;
  };

  std::vector<Triplet> triplets;
  for (PetscInt z = start.z(); z < end.z(); ++z) {
  for (PetscInt y = start.y(); y < end.y(); ++y) {
  for (PetscInt x = start.x(); x < end.x(); ++x) {
    PetscInt xm = (geom_nx > 1) ? (x - 1) : x;
    PetscInt ym = (geom_ny > 1) ? (y - 1) : y;
    PetscInt zm = (geom_nz > 1) ? (z - 1) : z;

    if (bool success = remap_with_boundaries(xm, ym, zm); !success)
      continue;

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
  return ((z * geom_ny + y) * geom_nx + x) * Vector3_dim + c;
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
  PetscCallVoid(VecDestroy(&E_));
  PetscCallVoid(VecDestroy(&B_));
  PetscCallVoid(VecDestroy(&J_));
  PetscCallVoid(MatDestroy(&rot_dt_p));
  PetscCallVoid(MatDestroy(&rot_dt_m));
  PetscFunctionReturnVoid();
}

}
