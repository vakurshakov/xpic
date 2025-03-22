#include "table_diagnostic.h"

TableDiagnostic::TableDiagnostic(const std::string& filename)
  : file_(filename)
{
}

PetscErrorCode TableDiagnostic::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t == 0) {
    PetscCall(initialize());
    PetscCall(add_titles());
    PetscCall(write_formatted("{:^15s}  ", titles_));
  }

  PetscCall(add_args());
  PetscCall(write_formatted("{: .6e}    ", args_));

  if (t % diagnose_period_ == 0)
    PetscCall(file_.flush());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void TableDiagnostic::add_arg(PetscReal arg, PetscInt pos)
{
  add(arg, args_, pos);
}

void TableDiagnostic::add_title(std::string title, PetscInt pos)
{
  add(title, titles_, pos);
}
