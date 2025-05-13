#ifndef SRC_COMMANDS_K_EQ_CYL_WITHOUT_PHI_H
#define SRC_COMMANDS_K_EQ_CYL_WITHOUT_PHI_H

#include "src/commands/inject_particles.h"
#include "src/utils/particles_load.h"
//
#include "table_function.h"
#include "ziggurat_gaussian.h"

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

  PetscReal a;
  TableFunction table_n;
  static constexpr PetscReal n0_tolerance = 1e-4;
};


class LoadMomentum {
public:
  LoadMomentum(SortParameters params, bool tov, std::string_view param_str);
  Vector3R operator()(const Vector3R& reference);

  void scale_coordinates(PetscReal scale);
  void scale_chi(PetscReal scale);

private:
  SortParameters params;
  bool tov;

  PetscReal a;
  TableFunction table_chi;

  ZigguratGaussian gauss;
};

}  // namespace cyl_without_phi
}  // namespace kotelnikov_equilibrium

#endif  // SRC_COMMANDS_K_EQ_CYL_WITHOUT_PHI_H
