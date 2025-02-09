#include "set_magnetic_field.h"

#include "src/utils/configuration.h"

SetMagneticField::SetMagneticField(Vec storage, Setter&& setup)
  : storage_(storage), setup_(std::move(setup))
{
}

PetscErrorCode SetMagneticField::execute(timestep_t /* t */)
{
  PetscFunctionBeginUser;
  PetscCall(setup_(storage_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


SetUniformField::SetUniformField(const Vector3R& value)
  : value_(value)
{
}

PetscErrorCode SetUniformField::operator()(Vec vec)
{
  PetscFunctionBeginUser;
  PetscCall(VecStrideSet(vec, X, value_[X]));
  PetscCall(VecStrideSet(vec, Y, value_[Y]));
  PetscCall(VecStrideSet(vec, Z, value_[Z]));
  LOG("  Magnetic fields is set: value ({} {} {}), energy {}",
    REP3_A(value_), 0.5 * value_.squared() * Vector3I{Geom_n}.elements_product());
  PetscFunctionReturn(PETSC_SUCCESS);
}
