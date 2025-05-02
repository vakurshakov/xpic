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
}

void LoadCoordinate::scale_coordinates(PetscReal scale)
{
  table_n.scale_coordinates(scale);
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
  while (np < n0_tolerance && random_01() > np);

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
  return ROUND_STEP(2.0 * M_PI * integral, dx * dy) * geom_nz;
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
  a = table_chi.get_xmin();

  ziggurat_generate_table();
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

  Vector3R v;
  ziggurat_generate_velocity(r, v);

  v[Z] = maxwell_inverse(random_01() / (2.0 * M_PI));

  // from thermal velocity units to `c`
  v *= std::sqrt(params.Tx / (params.m * mec2));

  if (!tov) {
    v *= params.m / std::sqrt(1.0 + v.length());
  }
  return v;
}

/// @note `v` here is in thermal velocity units
void LoadMomentum::ziggurat_generate_table()
{
  // `v` here is the approximation of the bottom slab width on the
  // distribution function, computed to satisfy the check below
  PetscReal v = 3.9235487686865955;
  PetscReal f = maxwell(v);

  const PetscReal A = M_PI * v * v * f + std::exp(-0.5 * v * v);

  table_v[0] = std::sqrt(A / (M_PI * f));
  table_f[0] = f;

  table_v[1] = v;
  table_f[1] = f;

  for (PetscInt i = 2; i < max_n; ++i) {
    table_v[i] = v = maxwell_inverse(f);
    table_f[i] = f = f + A / (M_PI * v * v);
  }

  PetscReal f0 = maxwell(0.0);
  PetscCheckAbort(std::abs(f - f0) < 1e-10, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "The resulting table is not precise enough, the value at the top of the ziggurat is %f, expected %f", f, f0);

  // Forcing the last point to be precisely (0, f(0))
  table_v[max_n - 1] = 0.0;
  table_f[max_n - 1] = f0;
}

/// @note `v` here is in thermal velocity units
void LoadMomentum::ziggurat_generate_velocity(PetscReal r, Vector3R& _v) const
{
  static std::uniform_int_distribution random_level(0, max_n - 1);

  // because the velocities are random, we do not track the "psi"-coordinate
  PetscReal& vx = _v[X];
  PetscReal& vy = _v[Y];

  const PetscReal chi_r = (r > a ? table_chi.get_value(r) : -1.0);
  const PetscReal chi_a = (r > a ? table_chi.get_value(a) : -1.0);

  PetscInt e = 0;
  for (; e < max_eval; ++e) {
    PetscInt i = random_level(RandomGenerator::get());
    PetscReal v_i = table_v[i];

    bool s1 = std::abs(v_i * a - (chi_r - chi_a)) > v_i * r;
    bool s2 = std::abs(v_i * a + (chi_r - chi_a)) > v_i * r;
    if (r > a && s1 && s2)
      continue;

    PetscReal v = v_i * std::sqrt(random_01());
    PetscReal theta = 2.0 * M_PI * random_01();

    vx = v * std::cos(theta);
    vy = v * std::sin(theta);

    if (r > a && std::abs(r * vy + chi_r) - (a * v + chi_a) < 0.0)
      continue;

    if (v < table_v[i + 1])
      return;

    if (i == 0) {
      do {
        v = std::sqrt(v_i * v_i - 2.0 * std::log(random_01()));
      }
      while (random_01() * v > v_i);

      vx = v * std::cos(theta);
      vy = v * std::sin(theta);
      return;
    }

    PetscReal f0 = table_f[i + 0];
    PetscReal f1 = table_f[i + 1];
    PetscReal f = f0 + random_01() * (f1 - f0);

    if (f < maxwell(v))
      return;
  }

  PetscCheckAbort(e == max_eval, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Maximum number of velocity evaluations %d is exceeded", max_eval);
}

/// @note `v` here is in thermal velocity units
/* static */ PetscReal LoadMomentum::maxwell(PetscReal v)
{
  return std::exp(-0.5 * v * v) / (2.0 * M_PI);  // f(v)
}

/// @note returns `v` in thermal velocity units
/* static */ PetscReal LoadMomentum::maxwell_inverse(PetscReal f)
{
  return std::sqrt(-2.0 * std::log(2.0 * M_PI * f));  // v(f)
}

}  // namespace cyl_without_phi
}  // namespace kotelnikov_equilibrium
