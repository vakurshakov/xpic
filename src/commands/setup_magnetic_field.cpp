#include "setup_magnetic_field.h"

#include "src/utils/configuration.h"

SetupMagneticField::SetupMagneticField(DM da, Vec storage, const Vector3R& value)
  : da_(da), storage_(storage), value_(value)
{
}

PetscErrorCode SetupMagneticField::execute(timestep_t /* t */)
{
  PetscFunctionBeginUser;
  Vector3R*** storage;
  PetscCall(DMDAVecGetArrayRead(da_, storage_, reinterpret_cast<void*>(&storage)));

  Vector3I start;
  Vector3I end;
  PetscCall(DMDAGetCorners(da_, REP3_A(&start), REP3_A(&end)));
  end += start;

#pragma omp parallel for simd
  // clang-format off
  for (PetscInt z = start.z(); z < end.z(); ++z) {
  for (PetscInt y = start.y(); y < end.y(); ++y) {
  for (PetscInt x = start.x(); x < end.x(); ++x) {
    storage[z][y][x] = value_;
  }}}
  // clang-format on
  PetscFunctionReturn(PETSC_SUCCESS);
}
