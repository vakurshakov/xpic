#ifndef SRC_DIAGNOSTICS_UTILS_TABLE_DIAGNOSTIC_H
#define SRC_DIAGNOSTICS_UTILS_TABLE_DIAGNOSTIC_H

#include "src/interfaces/diagnostic.h"
#include "src/utils/sync_file.h"

class TableDiagnostic : public interfaces::Diagnostic {
public:
  TableDiagnostic(const std::string& filename);
  PetscErrorCode diagnose(PetscInt t) override;

  virtual PetscErrorCode initialize();
  virtual PetscErrorCode add_columns(PetscInt t);

  template<typename T>
  using Format = std::format_string<T&>;

  template<typename T>
  void add(
    PetscInt w, std::string title, Format<T> fmt, T value, PetscInt pos = -1)
  {
    title = std::format("{:<{}.{}s}", title, w, w);

    std::string fvalue;
    fvalue = std::format(fmt, value);
    fvalue = std::format("{:^{}.{}s}", fvalue, w, w);

    if (pos >= 0) {
      titles_.insert(titles_.begin() + pos, title);
      values_.insert(values_.begin() + pos, fvalue);
    }
    else {
      titles_.push_back(title);
      values_.push_back(fvalue);
    }
  }

protected:
  PetscErrorCode write_formatted(const std::vector<std::string>& container);

  SyncFile file_;
  std::vector<std::string> titles_;
  std::vector<std::string> values_;
};

#endif  // SRC_DIAGNOSTICS_UTILS_TABLE_DIAGNOSTIC_H
