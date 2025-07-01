#ifndef SRC_DIAGNOSTICS_UTILS_TABLE_DIAGNOSTIC_H
#define SRC_DIAGNOSTICS_UTILS_TABLE_DIAGNOSTIC_H

#include "src/interfaces/diagnostic.h"
#include "src/utils/sync_file.h"

class TableDiagnostic : public interfaces::Diagnostic {
public:
  TableDiagnostic(const std::string& filename);
  PetscErrorCode diagnose(PetscInt t) override;

protected:
  virtual PetscErrorCode initialize();
  virtual PetscErrorCode add_titles() = 0;
  virtual PetscErrorCode add_args(PetscInt t) = 0;

  template<typename T>
  PetscErrorCode write_formatted(
    std::format_string<const T&> fmt, const std::vector<T>& container)
  {
    PetscFunctionBeginUser;
    for (PetscInt i = 0; i < (PetscInt)container.size() - 1; ++i) {
      file_() << std::format(fmt, container[i]);
    }

    auto last = std::format(fmt, container.back());
    while (last.back() == ' ') {
      last.pop_back();
    }

    file_() << last << "\n";
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  void add_arg(PetscReal arg, PetscInt pos = -1);
  void add_title(std::string title, PetscInt pos = -1);

  template<typename T>
  void add(T value, std::vector<T>& container, PetscInt pos = -1)
  {
    if (pos >= 0) {
      auto it = container.begin();
      std::advance(it, pos);
      container.insert(it, value);
    }
    else {
      container.push_back(value);
    }
  }

  SyncFile file_;
  std::vector<PetscReal> args_;
  std::vector<std::string> titles_;
};

#endif  // SRC_DIAGNOSTICS_UTILS_TABLE_DIAGNOSTIC_H
