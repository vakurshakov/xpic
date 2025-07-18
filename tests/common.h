#include "src/pch.h"
#include "src/interfaces/point.h"
#include "src/diagnostics/utils/table_diagnostic.h"

/// @todo There would be probably lots of utilities, they should be placed to some static library

bool equal_tol(PetscReal a, PetscReal b, PetscReal tol)
{
  return std::abs(a - b) < tol;
}

bool equal_tol(const Vector3R& a, const Vector3R& b, PetscReal tol)
{
  return  //
    equal_tol(a[X], b[X], tol) &&  //
    equal_tol(a[Y], b[Y], tol) &&  //
    equal_tol(a[Z], b[Z], tol);
}

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

      PetscCheck(equal_tol(e_n, o_n, PETSC_SMALL), PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Column values differ, at row %" PetscInt_FMT " col %" PetscInt_FMT " (%s) expected: %.6e, output: %.6e", row+2, col+1, titles[col].c_str(), e_n, o_n);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @warning For this to become a proper diagnostic,
/// dangling references to `point` have to be solved
class PointTrace : public TableDiagnostic {
public:
  PointTrace(std::string_view file, std::string_view id,  //
    const Point& point, PetscInt skip = 1)
    : TableDiagnostic(get_outputfile(file, id)), skip(skip), point(point)
  {
  }

private:
  PetscInt skip;
  const Point& point;

  PetscErrorCode add_titles() override
  {
    PetscFunctionBeginUser;
    add_title("t_[1/wpe]");
    add_title("x_[c/wpe]");
    add_title("y_[c/wpe]");
    add_title("z_[c/wpe]");
    add_title("vx_[c]");
    add_title("vy_[c]");
    add_title("vz_[c]");
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode add_args(PetscInt t) override
  {
    if (t % skip != 0)
      return PETSC_SUCCESS;

    PetscFunctionBeginUser;
    add_arg(t * dt);
    add_arg(point.x());
    add_arg(point.y());
    add_arg(point.z());
    add_arg(point.px());
    add_arg(point.py());
    add_arg(point.pz());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  std::filesystem::path get_outputfile(std::string_view file, std::string_view id)
  {
    return std::format("{}/temporal/{}.txt", get_out_dir(file).c_str(), id);
  }
};

class PointByFieldTrace : public TableDiagnostic {
public:
  PointByFieldTrace(std::string_view file, std::string_view id,  
    const PointByField& point, PetscInt skip = 1)
    : TableDiagnostic(get_outputfile(file, id)), skip(skip), point(point)
  {
  }

private:
  PetscInt skip;
  const PointByField& point;

  PetscErrorCode add_titles() override
  {
    PetscFunctionBeginUser;
    add_title("t_[1/wpe]");
    add_title("x_[c/wpe]");
    add_title("y_[c/wpe]");
    add_title("z_[c/wpe]");
    add_title("p_par_[mc]");
    add_title("p_perp_[mc]");
    add_title("mu_p_[]");    // или укажи единицы, если у тебя приняты (например, [mc^2/B])
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode add_args(PetscInt t) override
  {
    if (t % skip != 0)
      return PETSC_SUCCESS;

    PetscFunctionBeginUser;
    add_arg(t * dt);
    add_arg(point.x());
    add_arg(point.y());
    add_arg(point.z());
    add_arg(point.p_par());
    add_arg(point.p_perp_ref());
    add_arg(point.mu());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static std::filesystem::path get_outputfile(std::string_view file, std::string_view id)
  {
    return std::format("{}/temporal/{}.txt", get_out_dir(file).c_str(), id);
  }
};

void update_counter_clockwise(  //
  const Vector3R& old_r, const Vector3R& new_r,  //
  const Vector3R& B_p, PetscReal& counter_clockwise)
{
  Vector3R or_t = old_r.transverse_to(B_p);
  Vector3R nr_t = new_r.transverse_to(B_p);
  counter_clockwise += or_t.cross(nr_t).dot(B_p) / B_p.length();
}
