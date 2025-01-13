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

PetscErrorCode FieldsEnergy::diagnose(timestep_t t)
{
  PetscFunctionBeginUser;
  PetscCall(calculate_energies());
  PetscCall(file_.write_floats(3, energy_E_.data.data()));
  PetscCall(file_.write_floats(3, energy_B_.data.data()));

  PetscReal total =                              //
    energy_E_[X] + energy_E_[Y] + energy_E_[Z] + //
    energy_B_[X] + energy_B_[Y] + energy_B_[Z];

  PetscCall(file_.write_floats(1, &total));

  if (t % diagnose_period == 0)
    file_.flush();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldsEnergy::calculate_energies()
{
  PetscFunctionBeginUser;
  Vector3I start;
  Vector3I end;
  PetscCall(DMDAGetCorners(da_, REP3_A(&start), REP3_A(&end)));
  end += start;

  auto calculate_energy = [&](Vec f_, Vector3R& result) {
    PetscFunctionBeginUser;
    Vector3R*** f;
    PetscCall(DMDAVecGetArrayRead(da_, f_, reinterpret_cast<void*>(&f)));

    PetscReal fx = 0.0;
    PetscReal fy = 0.0;
    PetscReal fz = 0.0;

#pragma omp parallel for simd reduction(+ : fx, fy, fz)
    // clang-format off
    for (PetscInt z = start.z(); z < end.z(); ++z) {
    for (PetscInt y = start.y(); y < end.y(); ++y) {
    for (PetscInt x = start.x(); x < end.x(); ++x) {
      fx += f[z][y][x].x() * f[z][y][x].x();
      fy += f[z][y][x].y() * f[z][y][x].y();
      fz += f[z][y][x].z() * f[z][y][x].z();
    }}}
    // clang-format on

    result = 0.5 * Vector3R{fx, fy, fz} * (dx * dy * dz);
    PetscCall(DMDAVecRestoreArrayRead(da_, f_, reinterpret_cast<void*>(&f)));
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  PetscCall(calculate_energy(E_, energy_E_));
  PetscCall(calculate_energy(B_, energy_B_));

  constexpr PetscInt count = 6;
  PetscReal sendbuf[count];
  sendbuf[0] = energy_E_[X];
  sendbuf[1] = energy_E_[Y];
  sendbuf[2] = energy_E_[Z];
  sendbuf[3] = energy_B_[X];
  sendbuf[4] = energy_B_[Y];
  sendbuf[5] = energy_B_[Z];

  PetscReal recvbuf[count];
  PetscCallMPI(MPI_Reduce(sendbuf, recvbuf, count, MPIU_REAL, MPI_SUM, 0, PETSC_COMM_WORLD));

  energy_E_[X] = recvbuf[0];
  energy_E_[Y] = recvbuf[1];
  energy_E_[Z] = recvbuf[2];
  energy_B_[X] = recvbuf[3];
  energy_B_[Y] = recvbuf[4];
  energy_B_[Z] = recvbuf[5];
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
