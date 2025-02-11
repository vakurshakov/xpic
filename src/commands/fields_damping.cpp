#include "fields_damping.h"

#include "src/diagnostics/fields_energy.h"
#include "src/utils/configuration.h"

class FieldsDamping::DampForBox {
public:
  void operator()(
    PetscInt z, PetscInt y, PetscInt x, Vector3R*** f, PetscReal& energy);
  BoxGeometry geom_;
  PetscReal coefficient_;
};

FieldsDamping::FieldsDamping( //
  DM da, Vec E, Vec B, Vec B0, const BoxGeometry& geom, PetscReal coefficient)
  : da_(da), E_(E), B_(B), B0_(B0), damp_(DampForBox(geom, coefficient))
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

    damp_(z, y, x, arr, damped_energy_);
  }

  PetscCall(DMDAVecRestoreArrayWrite(da_, f, reinterpret_cast<void*>(&arr)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal FieldsDamping::get_damped_energy() const
{
  return damped_energy_;
}


void FieldsDamping::DampForBox::operator()(
  PetscInt z, PetscInt y, PetscInt x, Vector3R*** f, PetscReal& energy)
{
  const Vector3R r{
    x * dx,
    y * dy,
    z * dz,
  };

  bool is_outside = //
    (geom_.min[X] > r[X] || r[X] > geom_.max[X]) ||
    (geom_.min[Y] > r[Y] || r[Y] > geom_.max[Y]) ||
    (geom_.min[Z] > r[Z] || r[Z] > geom_.max[Z]);

  if (!is_outside)
    return;

  for (PetscInt i = 0; i < 3; ++i) {
    PetscReal width = 0.0;
    PetscReal delta = 0.0;

    if (r[i] > geom_.max[i]) {
      width = Geom[i] - geom_.max[i];
      delta = r[i] - geom_.max[i];
    }
    else if (r[i] < geom_.min[i]) {
      width = geom_.min[i] - 0;
      delta = r[i] - 0;
    }
    else
      continue;

    PetscReal damping = 1.0 - coefficient_ * POW2(delta / width - 1.0);

    energy += FieldsEnergy::get(f[z][y][x]) * (1.0 - POW2(damping));
    f[z][y][x] *= damping;
  }
}
