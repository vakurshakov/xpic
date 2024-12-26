#include "setup_magnetic_field.h"

#include "src/utils/configuration.h"

class SetupMagneticField::UniformField {
public:
  UniformField(const Vector3R& value);
  PetscErrorCode operator()(Vec storage);
  Vector3R value_;
};


SetupMagneticField::SetupMagneticField(Vec storage, const Vector3R& value)
  : storage_(storage), setup_(UniformField(value))
{
}

PetscErrorCode SetupMagneticField::execute(timestep_t /* t */)
{
  PetscFunctionBeginUser;
  PetscCall(setup_(storage_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


SetupMagneticField::UniformField::UniformField(const Vector3R& value)
  : value_(value)
{
}

PetscErrorCode SetupMagneticField::UniformField::operator()(Vec vec)
{
  PetscFunctionBeginUser;
  PetscCall(VecStrideSet(vec, X, value_[X]));
  PetscCall(VecStrideSet(vec, Y, value_[Y]));
  PetscCall(VecStrideSet(vec, Z, value_[Z]));
  PetscFunctionReturn(PETSC_SUCCESS);
}
