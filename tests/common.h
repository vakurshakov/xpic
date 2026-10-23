#include "src/pch.h"
#include "src/interfaces/point.h"
#include "src/diagnostics/particle_trace.h"

/// @todo There would be probably lots of utilities, they should be placed to some static library

bool equal_tol(PetscReal a, PetscReal b, PetscReal tol)
{
  return std::abs(a - b) < tol;
}

/// @brief Relative comparison with an absolute floor `atol`, so that the values
/// being zero up to the round-off noise are equal to each other.
bool equal_rel(PetscReal a, PetscReal b, PetscReal tol, PetscReal atol)
{
  return std::abs(a - b) <= tol * std::max(std::abs(a), std::abs(b)) + atol;
}

bool equal_tol(const Vector3R& a, const Vector3R& b, PetscReal tol)
{
  return                           //
    equal_tol(a[X], b[X], tol) &&  //
    equal_tol(a[Y], b[Y], tol) &&  //
    equal_tol(a[Z], b[Z], tol);
}

/// @brief Reads exactly `n` floats written by a field diagnostic (the whole file, on every rank).
PetscErrorCode read_floats(
  const std::filesystem::path& file, PetscInt n, std::vector<float>& out)
{
  PetscFunctionBeginUser;
  const uintmax_t expected_size = n * sizeof(float);
  PetscCheck(std::filesystem::file_size(file) == expected_size, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Result file size should match the region");

  out.resize(n);
  std::ifstream ifs(file, std::ios::binary);
  ifs.read(reinterpret_cast<char*>(out.data()), expected_size);
  PetscCheck(ifs.gcount() == static_cast<std::streamsize>(expected_size), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Result file could not be read completely");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Compares the file, covering the whole `geom_n*` grid, with the local part of `expected` living on `da`.
PetscErrorCode compare_with_vec(
  const std::filesystem::path& file, DM da, Vec expected, PetscReal tol)
{
  PetscFunctionBeginUser;
  PetscInt dof;
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetInfo(da, nullptr, REP3(nullptr), REP3(nullptr), &dof, nullptr, REP4(nullptr)));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

  std::vector<float> written;
  PetscCall(read_floats(file, geom_nx * geom_ny * geom_nz * dof, written));

  const PetscReal**** arr;
  PetscCall(DMDAVecGetArrayDOFRead(da, expected, &arr));
  for (PetscInt z = zs; z < zs + zm; ++z) {
    for (PetscInt y = ys; y < ys + ym; ++y) {
      for (PetscInt x = xs; x < xs + xm; ++x) {
        for (PetscInt c = 0; c < dof; ++c) {
          PetscCheck(equal_tol(written[indexing::t_g(x, y, z, c, dof)], arr[z][y][x][c], tol), PETSC_COMM_WORLD, PETSC_ERR_USER,
            "Written value differs from the expected one at (%d, %d, %d, %d)", x, y, z, c);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOFRead(da, expected, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

std::filesystem::path get_out_dir(std::string_view file)
{
  std::filesystem::path result(file);
  result.replace_extension("");

  result = std::format("{}/output/{}/",  //
    result.parent_path().c_str(), result.filename().c_str());
  return result;
}

/// @brief Compares the table `diag` with the expected one, relative error is
/// `PETSC_SMALL`. Use `atol` for columns that are only a round-off noise.
PetscErrorCode compare_temporal(std::string_view file, std::string_view diag,
  PetscReal atol = PETSC_MACHINE_EPSILON)
{
  PetscFunctionBeginUser;
  auto get_filepath = [&](std::string_view type)
  {
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

  auto get_linestream = [](std::istream& is)
  {
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
      "Incorrect header size of output, error occurred at col %d (%s)", col+1, e_title.c_str());

    PetscCheck(e_title == o_title, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "File headers must be the same, at col %d expected: \"%s\", output: \"%s\"", col+1, e_title.c_str(), o_title.c_str());

    titles.push_back(e_title);
  }

  PetscReal e_n;
  PetscReal o_n;

  for (PetscInt row = 0; (bool)e_ifs; ++row) {
    for (PetscInt col = 0; col < (PetscInt)titles.size(); ++col) {
      e_ifs >> e_n;
      o_ifs >> o_n;

      PetscCheck((bool)e_ifs == (bool)o_ifs, PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Incorrect filesize of output, error occurred at row %d col %d (%s)", row+2, col+1, titles[col].c_str());

      PetscCheck(equal_rel(e_n, o_n, PETSC_SMALL, atol), PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Column values differ, at row %d col %d (%s) expected: %.6e, output: %.6e", row+2, col+1, titles[col].c_str(), e_n, o_n);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @warning For this to become a proper diagnostic,
/// dangling references to `point` have to be solved
class PointTrace : public ParticleTrace<Point> {
public:
  PointTrace(std::string_view file, std::string_view id,  //
    const Point& point, PetscInt skip = 1)
    : ParticleTrace(
        std::format("{}/temporal/{}.txt", get_out_dir(file).c_str(), id),
        [&point]
        {
          return &point;
        },
        skip)
  {
  }
};

void update_counter_clockwise(                   //
  const Vector3R& old_r, const Vector3R& new_r,  //
  const Vector3R& B_p, PetscReal& counter_clockwise)
{
  Vector3R or_t = old_r.transverse_to(B_p);
  Vector3R nr_t = new_r.transverse_to(B_p);
  counter_clockwise += or_t.cross(nr_t).dot(B_p) / B_p.length();
}
