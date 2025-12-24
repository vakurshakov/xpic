#include "table_diagnostic.h"

TableDiagnostic::TableDiagnostic(const std::string& filename)
  : file_(filename)
{
}

PetscErrorCode TableDiagnostic::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t == 0)
    PetscCall(initialize());

  PetscCall(add_columns(t));

  if (!values_.empty()) {
    if (t == 0)
      PetscCall(write_formatted(titles_));
    PetscCall(write_formatted(values_));

    titles_.clear();
    values_.clear();
  }

  if (t % diagnose_period_ == 0)
    PetscCall(file_.flush());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TableDiagnostic::initialize()
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TableDiagnostic::add_columns(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode TableDiagnostic::write_formatted(
  const std::vector<std::string>& container)
{
  PetscFunctionBeginUser;
  PetscInt i = 0, size = (PetscInt)container.size();

  for (; i < size - 1; ++i) {
    file_() << container[i] << "  ";
  }

  auto last = container.back();
  while (!last.empty() && last.back() == ' ') {
    last.pop_back();
  }

  file_() << last << "\n";
  PetscFunctionReturn(PETSC_SUCCESS);
}
