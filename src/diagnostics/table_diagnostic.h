#ifndef SRC_DIAGNOSTICS_UTILS_TABLE_DIAGNOSTIC_H
#define SRC_DIAGNOSTICS_UTILS_TABLE_DIAGNOSTIC_H

#include "src/interfaces/diagnostic.h"

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
      titles.insert(titles.begin() + pos, title);
      values.insert(values.begin() + pos, fvalue);
    }
    else {
      titles.push_back(title);
      values.push_back(fvalue);
    }
  }

  void add_separator()
  {
    add(4, " |", "{}", "|");
  }

protected:
  PetscErrorCode write_formatted(const std::vector<std::string>& container);
  PetscErrorCode open(std::ios::openmode mode);
  PetscErrorCode flush();
  PetscErrorCode close();
  static bool is_synchronized();

  std::vector<std::string> titles;
  std::vector<std::string> values;

  const std::string filename;
  std::ofstream file;
  bool has_header = false;

  PetscInt event_write;
};

#endif  // SRC_DIAGNOSTICS_UTILS_TABLE_DIAGNOSTIC_H
