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

  std::filesystem::path e_path = get_filepath("expected");
  std::filesystem::path o_path = get_filepath("output");

  std::ifstream expected(e_path);
  std::ifstream output(o_path);

  PetscCheck(expected.is_open() == output.is_open(), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Both expected and output files must be open, filepaths:\n   expected: \"%s\"\n   output: \"%s\"", e_path.c_str(), o_path.c_str());

  std::string e_header;
  std::string o_header;

  std::getline(expected, e_header);
  std::getline(output, o_header);

  PetscCheck(e_header == o_header, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "File headers must be the same:\n   expected: \"%s\"\n   output: \"%s\"", e_header.c_str(), o_header.c_str());

  std::istringstream is_titles(e_header);
  std::vector<std::string> titles;

  for (std::string title; is_titles >> title;) {
    titles.push_back(title);
  }

  PetscReal e_n;
  PetscReal o_n;

  for (PetscInt row = 0; expected; ++row) {
    for (PetscInt col = 0; col < (PetscInt)titles.size(); ++col) {
      expected >> e_n;
      output >> o_n;

      PetscCheck(expected.fail() == output.fail(), PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Incorrect filesize of output, error occurred at row %" PetscInt_FMT " col %" PetscInt_FMT " (%s)", row+2, col+1, titles[col].c_str());

      PetscCheck(std::abs(e_n - o_n) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Column values differ, at row %" PetscInt_FMT " col %" PetscInt_FMT " (%s) expected: %.6e, output: %.6e", row+2, col+1, titles[col].c_str(), e_n, o_n);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
