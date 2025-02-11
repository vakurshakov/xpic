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

PetscErrorCode FieldsDamping::execute(timestep_t /* t */)
{
  PetscFunctionBeginUser;
  damped_energy_ = 0.0;

  PetscCall(VecAXPY(B_, -1.0, B0_));
  PetscCall(damping_implementation(E_));
  PetscCall(damping_implementation(B_));
  PetscCall(VecAXPY(B_, +1.0, B0_));

  LOG("  Fields are damped, additional energy runoff: {}", damped_energy_);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldsDamping::damping_implementation(Vec f)
{
  PetscFunctionBeginUser;
  Vector3I start;
  Vector3I size;
  PetscCall(DMDAGetCorners(da_, REP3_A(&start), REP3_A(&size)));

  Vector3R*** arr;
  PetscCall(DMDAVecGetArrayWrite(da_, f, reinterpret_cast<void*>(&arr)));

  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];
    const Vector3R r{x * dx, y * dy, z * dz};

    if (!within_geom_(r))
      damp_(r, arr[z][y][x], damped_energy_);
  }

  PetscCall(DMDAVecRestoreArrayWrite(da_, f, reinterpret_cast<void*>(&arr)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal FieldsDamping::get_damped_energy() const
{
  return damped_energy_;
}


void DampForBox::operator()(const Vector3R& r, Vector3R& f, PetscReal& energy)
{
  for (PetscInt i = 0; i < 3; ++i) {
    PetscReal width = 0.0;
    PetscReal delta = 0.0;

    if (r[i] > geom.max[i]) {
      width = Geom[i] - geom.max[i];
      delta = r[i] - geom.max[i];
    }
    else if (r[i] < geom.min[i]) {
      width = geom.min[i] - 0;
      delta = r[i] - 0;
    }
    else
      continue;

    PetscReal damping = 1.0 - coefficient * POW2(delta / width - 1.0);

    energy += FieldsEnergy::get(f) * (1.0 - POW2(damping));
    f *= damping;
  }
}
