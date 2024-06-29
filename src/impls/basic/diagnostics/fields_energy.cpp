#include "fields_energy.h"

#include "src/utils/utils.h"
#include "src/vectors/vector3.h"

namespace fs = std::filesystem;

namespace basic {

Fields_energy::Fields_energy(const std::string& out_dir, DM da, Vec E, Vec B)
  : interfaces::Diagnostic(out_dir), da_(da), E_(E), B_(B) {
  file_ = Sync_binary_file(out_dir_, "fields_energy");
}

PetscErrorCode Fields_energy::diagnose(timestep_t t) {
  PetscFunctionBeginUser;

  Vector3R ***E, ***B;
  PetscCall(DMDAVecGetArrayRead(da_, E_, &E));
  PetscCall(DMDAVecGetArrayRead(da_, B_, &B));

  Vector3I start, end;
  PetscCall(DMDAGetCorners(da_, REP3_A(&start), REP3_A(&end)));
  end += start;

  PetscReal WEx = 0.0, WEy = 0.0, WEz = 0.0;
  PetscReal WBx = 0.0, WBy = 0.0, WBz = 0.0;

  #pragma omp parallel for simd reduction(+: WEx, WEy, WEz, WBx, WBy, WBz)
  for (PetscInt z = start.z(); z < end.z(); ++z) {
  for (PetscInt y = start.y(); y < end.y(); ++y) {
  for (PetscInt x = start.x(); x < end.x(); ++x) {
    WEx += 0.5 * E[z][y][x].x() * E[z][y][x].x() * (dx * dy * dz);
    WEy += 0.5 * E[z][y][x].y() * E[z][y][x].y() * (dx * dy * dz);
    WEz += 0.5 * E[z][y][x].z() * E[z][y][x].z() * (dx * dy * dz);

    WBx += 0.5 * B[z][y][x].x() * B[z][y][x].x() * (dx * dy * dz);
    WBy += 0.5 * B[z][y][x].y() * B[z][y][x].y() * (dx * dy * dz);
    WBz += 0.5 * B[z][y][x].z() * B[z][y][x].z() * (dx * dy * dz);
  }}}

  PetscCall(DMDAVecRestoreArrayRead(da_, E_, &E));
  PetscCall(DMDAVecRestoreArrayRead(da_, B_, &B));

  auto write_reduced = [&](PetscReal& w) -> PetscErrorCode {
    PetscFunctionBeginUser;
    PetscReal sum = 0.0;
    PetscCallMPI(MPI_Reduce(&w, &sum, 1, MPIU_REAL, MPI_SUM, 0, PETSC_COMM_WORLD));

    w = sum; // only for logging
    PetscCall(file_.write_floats(1, &sum));
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  write_reduced(WEy);
  write_reduced(WEz);
  write_reduced(WBx);
  write_reduced(WBy);
  write_reduced(WBz);

  PetscReal total = WEx + WEy + WEz + WBx + WBy + WBz;
  PetscCall(file_.write_floats(1, &total));

  LOG_INFO("Fields energy: Ex = {:.5e}, Ey = {:.5e}, Ez = {:.5e}", WEx, WEy, WEz);
  LOG_INFO("               Bx = {:.5e}, By = {:.5e}, Bz = {:.5e}", WBx, WBy, WBz);
  LOG_INFO("            Total = {}", total);

  if (t % diagnose_period == 0) {
    file_.flush();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
