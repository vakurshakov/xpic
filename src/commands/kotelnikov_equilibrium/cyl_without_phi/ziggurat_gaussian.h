#ifndef SRC_COMMANDS_K_EQ_ZIGGURAT_GAUSSIAN_H
#define SRC_COMMANDS_K_EQ_ZIGGURAT_GAUSSIAN_H

#include "src/pch.h"

/**
 * @brief Kotelnikov's velocity distribution is essentially a standard
 * Maxwellian (Gaussian) distribution with the restrictions on possible
 * values of layer-parallel velocities.
 *
 * Satisfying those restriction, especially for larger values of r, will
 * require a lot of checks and re-generations of velocity components.
 * Standard implementation of gaussian distribution such as Box-Muller
 * transform uses computationally heavy `std::sqrt()` and `std::log()`
 * and, due to regenerations, this will degrade the performance of setup.
 *
 * To speed up the initialization, we will use the strategy called ziggurat
 * algorithm. Compared to the default approach, it uses rejection method
 * and, with tables of precomputed values, it can largely save time in
 * the scenario described above.
 *
 * For a detailed explanation of the method see
 *  Marsaglia, George, and Wai Wan Tsang.
 *  "The Ziggurat Method for Generating Random Variables",
 *  Journal of Statistical Software 5, no. 8 (2000).
 *  https://doi.org/10.18637/jss.v005.i08.
 */
class ZigguratGaussian {
public:
  ZigguratGaussian();

  PetscReal generate(PetscReal sigma = 1.0) const;

private:
  static void generate_table();

  static PetscReal gauss(PetscReal v);
  static PetscReal gauss_inverse(PetscReal f);

  static bool table_generated;

  static constexpr std::uint16_t max_n = 256;
  static std::uint32_t table_k[max_n];
  static PetscReal table_w[max_n];
  static PetscReal table_f[max_n];

  static constexpr PetscReal r = 3.6541528853610088;
};

#endif // SRC_COMMANDS_K_EQ_ZIGGURAT_GAUSSIAN_H
