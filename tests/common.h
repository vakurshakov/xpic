#include "src/pch.h"

/// @todo There would be probably lots of utilities, they should be placed to some static library

std::filesystem::path get_out_dir(std::string_view file)
{
  std::filesystem::path result(file);
  result.replace_extension("");

  result = std::format("{}/output/{}/",  //
    result.parent_path().c_str(), result.filename().c_str());
  return result;
}

PetscErrorCode compare_temporal(std::string_view file, std::string_view diag)
{
  PetscFunctionBeginUser;
  auto get_filepath = [&](std::string_view type) {
    std::filesystem::path result(file);
    result.replace_extension("");

    return std::format("{}/{}/{}/temporal/{}",  //
      result.parent_path().c_str(), type, result.filename().c_str(), diag);
  };

  auto e_path = get_filepath("expected");
  auto o_path = get_filepath("output");
  std::ifstream e_ifs(e_path);
  std::ifstream o_ifs(o_path);

  PetscCheck(e_ifs.is_open() == o_ifs.is_open(), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Both expected and output files must be open, filepaths:\n   expected: \"%s\"\n   output: \"%s\"", e_path.c_str(), o_path.c_str());

  auto get_linestream = [](std::istream& is) {
    std::string line;
    std::getline(is, line);
    return std::istringstream(line);
  };

  auto e_iss = get_linestream(e_ifs);
  auto o_iss = get_linestream(o_ifs);
  std::string e_title;
  std::string o_title;

  std::vector<std::string> titles;
  for (PetscInt col = 0; !e_iss.eof(); ++col) {
    e_iss >> e_title;
    o_iss >> o_title;

    PetscCheck((bool)e_iss == (bool)o_iss, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Incorrect header size of output, error occurred at col %" PetscInt_FMT " (%s)", col+1, e_title.c_str());

    PetscCheck(e_title == o_title, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "File headers must be the same, at col %" PetscInt_FMT " expected: \"%s\", output: \"%s\"", col+1, e_title.c_str(), o_title.c_str());

    titles.push_back(e_title);
  }

  PetscReal e_n;
  PetscReal o_n;

  for (PetscInt row = 0; (bool)e_ifs; ++row) {
    for (PetscInt col = 0; col < (PetscInt)titles.size(); ++col) {
      e_ifs >> e_n;
      o_ifs >> o_n;

      PetscCheck((bool)e_ifs == (bool)o_ifs, PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Incorrect filesize of output, error occurred at row %" PetscInt_FMT " col %" PetscInt_FMT " (%s)", row+2, col+1, titles[col].c_str());

      PetscCheck(std::abs(e_n - o_n) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Column values differ, at row %" PetscInt_FMT " col %" PetscInt_FMT " (%s) expected: %.6e, output: %.6e", row+2, col+1, titles[col].c_str(), e_n, o_n);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
