#include "mat_mult_field_view.h"

std::unique_ptr<MatMultFieldView> MatMultFieldView::create(
  const std::string& out_dir, DM result_da, Mat op, Vec source,
  const Region& region)
{
  PetscFunctionBeginUser;
  MPI_Comm newcomm;
  PetscCallAbort(PETSC_COMM_WORLD, World::create_local_comm(result_da, region, &newcomm));

  if (newcomm == MPI_COMM_NULL)
    PetscFunctionReturn(nullptr);

  Vec result;
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(result_da, &result));

  auto* diagnostic = new MatMultFieldView(  //
    out_dir, result_da, result, op, source, newcomm);
  PetscCallAbort(PETSC_COMM_WORLD, diagnostic->set_data_views(region));
  PetscFunctionReturn(std::unique_ptr<MatMultFieldView>(diagnostic));
}

MatMultFieldView::MatMultFieldView(const std::string& out_dir, DM result_da,
  Vec result, Mat op, Vec source, MPI_Comm newcomm)
  : FieldView(out_dir, result_da, result, newcomm), op(op), source(source)
{
}

PetscErrorCode MatMultFieldView::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&field));
  PetscCall(FieldView::finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultFieldView::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t % diagnose_period_ != 0)
    PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatMult(op, source, field));
  PetscCall(FieldView::diagnose(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}
