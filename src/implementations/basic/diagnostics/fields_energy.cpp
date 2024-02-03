#include "fields_energy.h"

#include "src/utils/utils.h"
#include "src/vectors/vector_classes.h"

namespace fs = std::filesystem;

namespace basic {

Fields_energy::Fields_energy(const std::string& result_directory, const DM da, const Vec E, const Vec B)
  : interfaces::Diagnostic(result_directory), da_(da), E_(E), B_(B) {
#if !START_FROM_BACKUP
  file_ = Binary_file(result_directory_, "fields_energy");
#else
  const int components_written = 7;
  file_ = Binary_file::from_backup(result_directory_, "fields_energy", components_written * sizeof(float));
#endif
}

PetscErrorCode Fields_energy::diagnose(timestep_t t) {
  PetscFunctionBeginUser;

  Vector3<PetscReal> ***E, ***B;
  PetscCall(DMDAVecGetArrayRead(da_, E_, &E));
  PetscCall(DMDAVecGetArrayRead(da_, B_, &B));

  Vector3<PetscInt> start, end;
  PetscCall(DMDAGetCorners(da_, R3DX(&start), R3DX(&end)));
  end += start;

  PetscReal WEx = 0.0, WEy = 0.0, WEz = 0.0;
  PetscReal WBx = 0.0, WBy = 0.0, WBz = 0.0;

  #pragma omp parallel for simd reduction(+: WEx, WEy, WEz, WBx, WBy, WBz)
  for (PetscInt z = start.z; z < end.z; ++z) {
  for (PetscInt y = start.y; y < end.y; ++y) {
  for (PetscInt x = start.x; x < end.x; ++x) {
    WEx += 0.5 * E[z][y][x].x * E[z][y][x].x * (dx * dy * dz);
    WEy += 0.5 * E[z][y][x].y * E[z][y][x].y * (dx * dy * dz);
    WEz += 0.5 * E[z][y][x].z * E[z][y][x].z * (dx * dy * dz);

    WBx += 0.5 * B[z][y][x].x * B[z][y][x].x * (dx * dy * dz);
    WBy += 0.5 * B[z][y][x].y * B[z][y][x].y * (dx * dy * dz);
    WBz += 0.5 * B[z][y][x].z * B[z][y][x].z * (dx * dy * dz);
  }}}

  PetscCall(DMDAVecRestoreArrayRead(da_, E_, &E));
  PetscCall(DMDAVecRestoreArrayRead(da_, B_, &B));

  file_.write_float(WEx);
  file_.write_float(WEy);
  file_.write_float(WEz);
  file_.write_float(WBx);
  file_.write_float(WBy);
  file_.write_float(WBz);

  PetscReal total = WEx + WEy + WEz + WBx + WBy + WBz;
  file_.write_float(total);

  LOG_INFO("Fields energy: Ex = {:.5e}, Ey = {:.5e}, Ez = {:.5e}", WEx, WEy, WEz);
  LOG_INFO("               Bx = {:.5e}, By = {:.5e}, Bz = {:.5e}", WBx, WBy, WBz);
  LOG_INFO("            Total = {}", total);

  if (t % diagnose_period == 0) {
    file_.flush();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
