#include "fields_damping.h"

#include "src/diagnostics/fields_energy.h"

FieldsDamping::FieldsDamping(
  DM da, Vec E, Vec B, Vec B0, Tester&& test, Damping&& damp)
  : da_(da),
    E_(E),
    B_(B),
    B0_(B0),
    within_geom_(std::move(test)),
    damp_(std::move(damp))
{
}

PetscErrorCode FieldsDamping::execute(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  damped_energy_ = 0.0;

  PetscCall(VecAXPY(B_, -1.0, B0_));
  PetscCall(damping_implementation(E_));
  PetscCall(damping_implementation(B_));
  PetscCall(VecAXPY(B_, +1.0, B0_));

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &damped_energy_, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  LOG("  Fields are damped, additional energy runoff: {:6.4e}", damped_energy_);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldsDamping::damping_implementation(Vec f)
{
  PetscFunctionBeginUser;
  Vector3I start;
  Vector3I size;
  PetscCall(DMDAGetCorners(da_, REP3_A(&start), REP3_A(&size)));

  Vector3R*** arr;
  PetscCall(DMDAVecGetArrayWrite(da_, f, &arr));

#pragma omp parallel for reduction(+ : damped_energy_)
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];
    Vector3R r{(x + 0.5) * dx, (y + 0.5) * dy, (z + 0.5) * dz};

    if (within_geom_(r))
      continue;

    PetscReal damping = damp_(r);
    Vector3R& f = arr[z][y][x];

    damped_energy_ += FieldsEnergy::get(f) * (1.0 - POW2(damping));
    f *= damping;
  }

  PetscCall(DMDAVecRestoreArrayWrite(da_, f, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal FieldsDamping::get_damped_energy() const
{
  return damped_energy_;
}


PetscReal DampForBox::operator()(const Vector3R& g)
{
  PetscReal damping = 1.0;

  for (PetscInt i = 0; i < 3; ++i) {
    PetscReal width = 0.0;
    PetscReal delta = 0.0;

    if (g[i] > geom.max[i]) {
      width = Geom[i] - geom.max[i];
      delta = g[i] - geom.max[i];
    }
    else if (g[i] < geom.min[i]) {
      width = geom.min[i] - 0;
      delta = g[i] - 0;
    }
    else
      continue;

    damping *= 1.0 - coefficient * POW2(delta / width - 1.0);
  }
  return damping;
}

PetscReal DampForCylinder::operator()(const Vector3R& g)
{
  PetscReal r = std::hypot(g[X] - geom.center[X], g[Y] - geom.center[Y]);

  /// @note Here we can be with r > geom.radius, @see `WithinCylinder::operator()`
  if (r < geom.radius)
    return 1.0;

  PetscReal width = geom.center[X] - geom.radius;
  PetscReal delta = r - geom.radius;
  PetscReal delta0 = width * (1.0 + 1.0 / std::sqrt(coefficient));

  /// @note To avoid negative damping coefficients, we check against `delta0`.
  PetscReal damping = 0.0;

  if (delta < delta0)
    damping = 1.0 - coefficient * POW2(delta / width - 1.0);

  return damping;
}
