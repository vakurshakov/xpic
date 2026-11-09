#include "table_diagnostic.h"

#include "src/utils/profiler.h"

TableDiagnostic::TableDiagnostic(const std::string& filename)
  : filename(filename)
{
}

PetscErrorCode TableDiagnostic::initialize()
{
  PetscFunctionBeginUser;
  initialized = true;
  PetscCall(Profiler::register_event("table_write", &event_write));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TableDiagnostic::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(Profiler::push_event(event_write));

  // The file is opened here, after `SimulationBackup` has loaded the temporal tables.
  // The run continuing from `t != 0` appends to them, the loaded table already has the row `t`.
  bool row_is_loaded = false;

  if (!initialized) {
    PetscCall(open(std::ios::out | (t == 0 ? std::ios::trunc : std::ios::app)));
    PetscCall(initialize());
    row_is_loaded = (t != 0 && has_header);
  }

  PetscCall(add_columns(t));

  if (!values.empty()) {
    if (!has_header) {
      has_header = true;
      PetscCall(write_formatted(titles));
    }

    if (!row_is_loaded) {
      PetscCall(write_formatted(values));
    }

    titles.clear();
    values.clear();
  }

  if (t % diagnose_period_ == 0)
    PetscCall(flush());

  PetscCall(Profiler::pop_event(event_write));
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
    file << container[i] << "  ";
  }

  auto last = container.back();
  while (!last.empty() && last.back() == ' ') {
    last.pop_back();
  }

  file << last << "\n";
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TableDiagnostic::open(std::ios::openmode mode)
{
  if (!is_synchronized())
    return PETSC_SUCCESS;

  PetscFunctionBeginHot;
  PetscCall(close());

  std::filesystem::path path(filename);
  std::filesystem::create_directories(path.parent_path());

  PetscCallCXX(file.open(path, mode));
  has_header = (std::filesystem::file_size(path) > 0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TableDiagnostic::flush()
{
  if (!is_synchronized())
    return PETSC_SUCCESS;

  PetscFunctionBeginHot;
  PetscCallCXX(file.flush());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TableDiagnostic::close()
{
  if (!is_synchronized())
    return PETSC_SUCCESS;

  PetscFunctionBeginHot;
  if (file.is_open()) {
    PetscCallCXX(file.flush());
    PetscCallCXX(file.close());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

bool TableDiagnostic::is_synchronized()
{
  PetscMPIInt flag;
  PetscCallMPI(MPI_Initialized(&flag));
  if (!static_cast<bool>(flag))
    return true;

  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0)
    return true;

  return false;
}
