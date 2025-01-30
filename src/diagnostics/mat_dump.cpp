#include "mat_dump.h"


MatDump::MatDump(const std::string& out_dir, Mat mat, const std::string& comp_dir)
  : interfaces::Diagnostic(out_dir), mat_(mat), comp_dir_(comp_dir)
{
}

PetscErrorCode MatDump::diagnose(timestep_t t)
{
  if (t % diagnose_period != 0)
    return PETSC_SUCCESS;
  PetscFunctionBeginUser;

  const std::string output = out_dir_ + "/" + format_time(t);
  std::filesystem::create_directories(out_dir_);

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, output.c_str(), FILE_MODE_WRITE, &viewer_));
  PetscCall(MatView(mat_, viewer_));
  PetscCall(PetscViewerDestroy(&viewer_));

  if (!comp_dir_.empty())
    compare(t);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDump::compare(timestep_t t)
{
  PetscFunctionBeginUser;
  Mat comp_mat;
  PetscCall(MatDuplicate(mat_, MAT_DO_NOT_COPY_VALUES, &comp_mat));

  const std::string comp_output = comp_dir_ + "/" + format_time(t);
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, comp_output.c_str(), FILE_MODE_READ, &viewer_));
  PetscCall(MatLoad(comp_mat, viewer_));
  PetscCall(PetscViewerDestroy(&viewer_));

  PetscReal norm;
  PetscCall(MatAXPY(comp_mat, -1.0, mat_, UNKNOWN_NONZERO_PATTERN));
  PetscCall(MatNorm(comp_mat, NORM_1, &norm));
  PetscCall(MatDestroy(&comp_mat));
  LOG("  MatDump::compare({}): norm of the difference in matrices {}", t, norm);
  PetscFunctionReturn(PETSC_SUCCESS);
}