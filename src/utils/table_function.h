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

  PetscReal get_xmin() const;
  PetscReal get_xmax() const;
  PetscReal get_dx() const;

  void scale_coordinates(PetscReal scale);
  void scale_values(PetscReal scale);

  /// @param x Coordinate to find a function value at.
  /// @return Linearly interpolated value of a stored parameter function.
  PetscReal get_value(PetscReal x) const;

private:
  PetscReal linear_interpolation(PetscReal v0, PetscReal v1, PetscReal t) const;

  PetscReal xmin_;  // - Start coordinate
  PetscReal xmax_;  // - Last coordinate
  PetscReal dx_;    // - Grid spacing

  std::vector<PetscReal> values_;
};

#endif  // SRC_COMMANDS_KOTELNIKOV_EQUILIBRIUM_TABLE_FUNCTION_H
