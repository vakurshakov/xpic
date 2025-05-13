#include "ziggurat_gaussian.h"

#include "src/utils/random_generator.h"

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
