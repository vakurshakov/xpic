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
  return 2 * ROUND_STEP(2 * M_PI * integral, dx * dy) * geom_nz;
}

PetscReal LoadCoordinate::get_probability(PetscReal r) const
{
  if (r <= table_n.get_xmin())
    return 1.0;
  else if (r <= table_n.get_xmax())
    return table_n.get_value(r);
  return 0.0;
}


bool ZigguratGaussian::table_generated = false;
std::uint32_t ZigguratGaussian::table_k[];
PetscReal ZigguratGaussian::table_w[];
PetscReal ZigguratGaussian::table_f[];

ZigguratGaussian::ZigguratGaussian()
{
  if (!table_generated)
    generate_table();
}

void ZigguratGaussian::generate_table()
{
  PetscReal x = r;
  PetscReal f = gauss(x);

  const PetscReal v = x * f + std::sqrt(M_PI_2) * std::erfc(x * M_SQRT1_2);
  const PetscReal p = std::pow(2.0, 32.0);

  table_k[0] = (std::uint32_t)((x * f / v) * p);
  table_k[1] = 0;

  table_w[0] = (v / f) / p;
  table_w[255] = x / p;

  table_f[0] = 1.0;
  table_f[255] = f;

  PetscReal xn;
  for (PetscInt i = 254; i >= 1; --i) {
    xn = x;
    x = gauss_inverse(v / x + f);
    f = gauss(x);
    table_k[i + 1] = (std::uint32_t)((x / xn) * p);
    table_w[i] = x / p;
    table_f[i] = f;
  }

#pragma omp atomic write
  table_generated = true;
}

PetscReal ZigguratGaussian::generate(PetscReal sigma) const
{
  // This will generate values in [0, 2^32)
  static std::uniform_int_distribution<std::uint32_t> random_i;

  PetscReal x, f;
  do {
    std::uint32_t i = random_i(RandomGenerator::get());
    std::uint8_t j = i & 255;

    x = i * table_w[j];
    if (i < table_k[j])
      break;

    if (j == 0) {
      do {
        x = -std::log(random_01()) / r;
        f = -std::log(random_01());
      }
      while (2.0 * f < x * x);

      x += r;
      break;
    }

    PetscReal f0 = table_f[j + 0];
    PetscReal f1 = table_f[j + 1];
    f = f1 + random_01() * (f0 - f1);
  }
  while (f >= gauss(x));

  return x * random_sign() * sigma;
}

PetscReal ZigguratGaussian::gauss(PetscReal x)
{
  return std::exp(-0.5 * x * x);
}

PetscReal ZigguratGaussian::gauss_inverse(PetscReal f)
{
  return std::sqrt(-2.0 * std::log(f));
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

  const PetscReal chi_r = (r > a ? table_chi.get_value(r) : -1.0);
  const PetscReal chi_a = (r > a ? table_chi.get_value(a) : -1.0);

  Vector3R v;
  PetscReal vn;
  do {
    v[R] = gauss.generate();
    v[A] = gauss.generate();
    vn = v.length();
  }
  while (r > a && std::abs(r * v[A] + chi_r) > (a * vn + chi_a));

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
