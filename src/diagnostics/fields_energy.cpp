#include "fields_energy.h"

#include "src/utils/utils.h"


FieldsEnergy::FieldsEnergy(DM da, Vec E, Vec B)
  : da_(da), E_(E), B_(B)
{
}

FieldsEnergy::FieldsEnergy(const std::string& out_dir, DM da, Vec E, Vec B)
  : interfaces::Diagnostic(out_dir),
    file_(SyncBinaryFile(out_dir_ + "/fields_energy.bin")),
    da_(da),
    E_(E),
    B_(B)
{
}

PetscErrorCode FieldsEnergy::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(calculate_energies());
  PetscCall(file_.write_floats(3, energy_E_.data));
  PetscCall(file_.write_floats(3, energy_B_.data));

  PetscReal total = get_electric_energy() + get_magnetic_energy();
  PetscCall(file_.write_floats(1, &total));

  if (t % diagnose_period == 0)
    file_.flush();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldsEnergy::calculate_energies()
{
  PetscFunctionBeginUser;
  PetscCall(VecStrideNormAll(E_, NORM_2, energy_E_));
  PetscCall(VecStrideNormAll(B_, NORM_2, energy_B_));
  energy_E_ = 0.5 * energy_E_.elementwise_product(energy_E_);
  energy_B_ = 0.5 * energy_B_.elementwise_product(energy_B_);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal FieldsEnergy::get_electric_energy() const
{
  return energy_E_.elements_sum();
}

PetscReal FieldsEnergy::get_magnetic_energy() const
{
  return energy_B_.elements_sum();
}

/* static */ PetscReal FieldsEnergy::get(const Vector3R& f)
{
  return 0.5 * f.squared();
}
