#include "simulation.h"

#include "src/utils/operators.h"
#include "src/utils/utils.h"


namespace ricketson {

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscMPIInt comm_size;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &comm_size));
  PetscCheck(comm_size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Ricketson scheme is uniprocessor currently");

  PetscCall(DMCreateGlobalVector(world.da, &E_));
  PetscCall(DMCreateGlobalVector(world.da, &B_));
  PetscCall(setup_norm_gradient());

  /// @todo Move this into class Set_approximate_magnetic_mirror.
  const PetscReal current = 3e4;
  const PetscReal radius = 3;
  const PetscReal distance = 15;

  Vector3I start;
  Vector3I size;
  PetscCall(DMDAGetCorners(world.da, REP3_A(&start), REP3_A(&size)));

  Vector3R*** B;
  PetscCall(DMDAVecGetArrayWrite(world.da, B_, &B));

  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    // clang-format off
    PetscReal B0 = 0.5 * current * POW2(radius) *
      (1 / pow(POW2(radius) + POW2((z * dz - 0.5 * geom_z) + 0.5 * distance), 1.5) +
        1 / pow(POW2(radius) + POW2((z * dz - 0.5 * geom_z) - 0.5 * distance), 1.5));
    // clang-format on

    PetscReal B1 = ((z * dz - 0.5 * geom_z) + 0.5 * distance) /
        (POW2(radius) + POW2((z * dz - 0.5 * geom_z) + 0.5 * distance)) +
      ((z * dz - 0.5 * geom_z) - 0.5 * distance) /
        (POW2(radius) + POW2((z * dz - 0.5 * geom_z) - 0.5 * distance));

    B[z][y][x].x() = B0 * 1.5 * (x * dx - 0.5 * geom_x) * B1;
    B[z][y][x].y() = B0 * 1.5 * (y * dy - 0.5 * geom_y) * B1;
    B[z][y][x].z() = B0 * 1.0;
  }
  PetscCall(DMDAVecRestoreArrayWrite(world.da, B_, &B));

  SortParameters parameters = {
    .sort_name = "positron",
    .Np = 1,
    .n = +1.0,
    .q = +1.0,
    .m = +1.0,
  };
  auto& sort = particles_.emplace_back(std::make_unique<Particles>(*this, parameters));

  const PetscReal v_crit = std::sqrt(9.8342 - 1);
  const PetscReal factor = 0.3;

  Vector3R r = {0, 0.02, 0};
  Vector3R v = {1, 0, factor * v_crit};

  sort->add_particle(Point{r, v});
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::calculate_b_norm_gradient()
{
  PetscFunctionBeginUser;
  Vector3R*** B;
  PetscCall(DMDAVecGetArrayRead(world.da, B_, &B));

  PetscReal* B_norm;
  PetscCall(VecGetArrayWrite(B_norm_, &B_norm));

  Vector3I start;
  Vector3I size;
  PetscCall(DMDAGetCorners(world.da, REP3_A(&start), REP3_A(&size)));

  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];
    B_norm[g] = B[z][y][x].length();
  }

  PetscCall(DMDAVecRestoreArrayRead(world.da, B_, &B));
  PetscCall(VecRestoreArrayWrite(B_norm_, &B_norm));

  PetscCall(MatMult(norm_gradient_, B_norm_, DB_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::setup_norm_gradient()
{
  PetscFunctionBeginUser;
  DM da = world.da;
  PetscCall(DMCreateGlobalVector(da, &DB_));

  PetscInt start[3];
  PetscInt size[3];
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  VecType vtype;
  PetscCall(DMGetVecType(da, &vtype));

  PetscInt ls = size[X] * size[Y] * size[Z];
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da), &B_norm_));
  PetscCall(VecSetSizes(B_norm_, ls, PETSC_DETERMINE));
  PetscCall(VecSetType(B_norm_, vtype));
  PetscCall(VecSetUp(B_norm_));

  /// @note The following is the setup of \vec{∇-} operator for |\vec{B}|.
  /// By defining negative derivative on Yee stencil, we set |\vec{B}|
  /// in (i+0.5, j+0.5, k+0.5).
  Gradient gradient(da);
  PetscCall(gradient.create_negative(&norm_gradient_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::timestep_implementation(PetscInt /* timestep */)
{
  PetscFunctionBeginUser;

  PetscCall(calculate_b_norm_gradient());

  for (auto& sort : particles_)
    PetscCall(sort->push());

  PetscFunctionReturn(PETSC_SUCCESS);
}


Vec Simulation::get_named_vector(std::string_view name)
{
  if (name == "E")
    return E_;
  if (name == "B")
    return B_;
  if (name == "DB")
    return DB_;
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
  PetscCallVoid(VecDestroy(&DB_));
  PetscCallVoid(VecDestroy(&B_norm_));  // dof = 1
  PetscCallVoid(MatDestroy(&norm_gradient_));
  PetscFunctionReturnVoid();
}

}  // namespace ricketson
