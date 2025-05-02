#ifndef SRC_COMMANDS_K_EQ_CYL_WITHOUT_PHI_H
#define SRC_COMMANDS_K_EQ_CYL_WITHOUT_PHI_H

#include "src/commands/inject_particles.h"
#include "src/utils/particles_load.h"
#include "table_function.h"

namespace kotelnikov_equilibrium {
namespace cyl_without_phi {

class SetEquilibriumField {
public:
  SetEquilibriumField(std::string_view param_str);
  PetscErrorCode operator()(Vec vec);

  void scale_coordinates(PetscReal scale);
  void scale_b(PetscReal scale);

private:
  TableFunction table_b;
};


class LoadCoordinate {
public:
  LoadCoordinate(std::string_view param_str);
  Vector3R operator()();
  PetscInt get_cells_number() const;

  void scale_coordinates(PetscReal scale);

private:
  PetscReal get_probability(PetscReal r) const;

  TableFunction table_n;
  static constexpr PetscReal n0_tolerance = 1e-3;
};

/**
 * @brief As the loading of Kotelnikov's velocity distribution requires
 * a _lot_ of checks and heavy `std::exp()`, `std::log()` calculations
 * during the setup, we will utilize the strategy called ziggurat algorithm.
 *
 * Compared to the default Box-Muller transform, it allows precomputed
 * table of Maxwellian distribution function values to be utilized and
 * the necessary check to be embedded into the process of velocity
 * generation to fasten the calculations.
 *
 * For reference see https://en.wikipedia.org/wiki/Ziggurat_algorithm.
 */
class LoadMomentum {
public:
  LoadMomentum(SortParameters params, bool tov, std::string_view param_str);
  Vector3R operator()(const Vector3R& reference);

  void scale_coordinates(PetscReal scale);
  void scale_chi(PetscReal scale);

private:
  SortParameters params;

  void ziggurat_generate_table();
  void ziggurat_generate_velocity(PetscReal r, Vector3R& v) const;

  static PetscReal maxwell(PetscReal v);
  static PetscReal maxwell_inverse(PetscReal f);

  static constexpr PetscInt max_n = 256 + 1;
  static constexpr PetscInt max_eval = 1000;
  PetscReal table_v[max_n];
  PetscReal table_f[max_n];

  PetscReal a;
  TableFunction table_chi;

  bool tov;
};

}  // namespace cyl_without_phi
}  // namespace kotelnikov_equilibrium

#endif  // SRC_COMMANDS_K_EQ_CYL_WITHOUT_PHI_H
