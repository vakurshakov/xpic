#include "set_cyl_without_phi.h"

#include "src/utils/random_generator.h"

namespace kotelnikov_equilibrium {
namespace cyl_without_phi {

namespace {

static std::string get_distribution(
  std::string_view param_str, std::string_view name)
{
  std::filesystem::path result(__FILE__);
  result = result.parent_path();
  result = result / "cache" / param_str / ("maxw_" + std::string(name));
  return result.string();
}

}

SetEquilibriumField::SetEquilibriumField(std::string_view param_str)
{
  PetscCallAbort(PETSC_COMM_WORLD, table_b.evaluate_from_file(get_distribution(param_str, "b")));
}

void SetEquilibriumField::scale_coordinates(PetscReal scale)
{
  table_b.scale_coordinates(scale);
}

void SetEquilibriumField::scale_b(PetscReal scale)
{
  table_b.scale_values(scale);
}

PetscErrorCode SetEquilibriumField::operator()(Vec vec)
{
  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Vector3R*** arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));

  PetscReal sx, sy, r;

  PetscReal rmin = table_b.get_xmin();
  PetscReal rmax = table_b.get_xmax();

#pragma omp parallel for private(sx, sy, r)
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    sx = (x + 0.5) * dx - 0.5 * geom_x;
    sy = (y + 0.5) * dy - 0.5 * geom_y;
    r = std::hypot(sx, sy);

    if (r < rmin)
      r = rmin;

    if (r > rmax)
      r = rmax;

    arr[z][y][x][Z] = table_b.get_value(r);
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, vec, &arr));

  LOG("  Magnetic field for Kotelnikov equilibrium is set!");
  PetscFunctionReturn(PETSC_SUCCESS);
}


LoadCoordinate::LoadCoordinate(std::string_view param_str)
{
  PetscCallAbort(PETSC_COMM_WORLD, table_n.evaluate_from_file(get_distribution(param_str, "n")));
  a = table_n.get_xmin();
}

void LoadCoordinate::scale_coordinates(PetscReal scale)
{
  table_n.scale_coordinates(scale);
  a *= scale;
}

Vector3R LoadCoordinate::operator()()
{
  PetscReal r;
  PetscReal np;
  PetscReal rmax = table_n.get_xmax();

  do {
    r = rmax * std::sqrt(random_01());
    np = get_probability(r);
  }
  while (r > a && (np < n0_tolerance || random_01() > np));

  PetscReal phi = 2.0 * M_PI * random_01();

  return Vector3R{
    0.5 * geom_x + r * std::cos(phi),
    0.5 * geom_y + r * std::sin(phi),
    geom_nz * random_01(),
  };
}

PetscInt LoadCoordinate::get_cells_number() const
{
  PetscReal dr = table_n.get_dx();
  PetscReal rmax = table_n.get_xmax();
  PetscReal integral = 0.0;

  for (PetscReal r = 0; r < rmax; r += dr) {
    if (PetscReal np = get_probability(r); np > n0_tolerance)
      integral += np * r * dr;
  }
  return ROUND_STEP(2 * M_PI * integral, dx * dy) * geom_nz;
}

PetscReal LoadCoordinate::get_probability(PetscReal r) const
{
  if (r <= table_n.get_xmin())
    return 1.0;
  else if (r <= table_n.get_xmax())
    return table_n.get_value(r);
  return 0.0;
}


LoadMomentum::LoadMomentum(
  SortParameters params, bool tov, std::string_view param_str)
  : params(params), tov(tov)
{
  PetscCallAbort(PETSC_COMM_WORLD, table_chi.evaluate_from_file(get_distribution(param_str, "chi")));
  table_chi.scale_values(M_SQRT1_2);

  a = table_chi.get_xmin();
}

void LoadMomentum::scale_coordinates(PetscReal scale)
{
  table_chi.scale_coordinates(scale);
  a *= scale;
}

void LoadMomentum::scale_chi(PetscReal scale)
{
  table_chi.scale_values(scale);
}

Vector3R LoadMomentum::operator()(const Vector3R& reference)
{
  const PetscReal x = reference[X] - 0.5 * geom_x;
  const PetscReal y = reference[Y] - 0.5 * geom_y;
  const PetscReal r = std::hypot(x, y);

  const PetscReal chi =
    r > a ? (table_chi.get_value(r) - table_chi.get_value(a)) : -1.0;

  Vector3R v;
  PetscReal vn;
  do {
    v[R] = gauss.generate();
    v[A] = gauss.generate();
    vn = v.length();
  }
  while (r > a && !(-a * vn < r * v[A] + chi && r * v[A] + chi < +a * vn));

  v[Z] = gauss.generate();

  // from thermal velocity units to `c`
  v *= std::sqrt(params.Tx / (params.m * mec2));

  if (!tov) {
    v *= params.m / std::sqrt(1.0 + v.length());
  }

  // Particles close to r=0 are not taken into account
  if (std::isinf(1.0 / r))
    return v;

  return {
    (+x * v[R] - y * v[A]) / r,
    (+y * v[R] + x * v[A]) / r,
    v[Z],
  };
}

}  // namespace cyl_without_phi
}  // namespace kotelnikov_equilibrium
