#ifndef SRC_COMMANDS_KOTELNIKOV_EQUILIBRIUM_TABLE_FUNCTION_H
#define SRC_COMMANDS_KOTELNIKOV_EQUILIBRIUM_TABLE_FUNCTION_H

#include "src/pch.h"

/**
 * @brief Carries the table-function values.
 *
 * @details It provides a table function that can be used via
 * `get_value()` method. The function gets the values from the binary
 * files. Linear interpolation is used for intermediate `x` values.
 */
class TableFunction {
public:
  TableFunction() = default;
  TableFunction(const std::string& filename);
  PetscErrorCode evaluate_from_file(const std::string& filename);

  // clang-format off
  PetscReal get_xmin() const { return xmin_; }
  PetscReal get_xmax() const { return xmax_; }
  PetscReal get_dx() const { return dx_; }
  // clang-format on

  void scale_coordinates(PetscReal scale);
  void scale_values(PetscReal scale);

  /// @param x Coordinate to find a function value at (in c/w_pe units).
  /// @return Linearly interpolated value of a stored parameter function.
  PetscReal get_value(PetscReal x) const;

private:
  PetscReal linear_interpolation(PetscReal v0, PetscReal v1, PetscReal t) const;

  PetscReal xmin_;  // [c/w_pe] - Start coordinate
  PetscReal xmax_;  // [c/w_pe] - Last coordinate
  PetscReal dx_;    // [c/w_pe] - Grid spacing

  std::vector<PetscReal> values_;
};

#endif  // SRC_COMMANDS_KOTELNIKOV_EQUILIBRIUM_TABLE_FUNCTION_H
