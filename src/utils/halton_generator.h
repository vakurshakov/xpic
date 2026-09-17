#ifndef SRC_UTILS_HALTON_GENERATOR_H
#define SRC_UTILS_HALTON_GENERATOR_H

#include <array>
#include <cstddef>

#include <petscsystypes.h>

namespace halton {

inline constexpr std::array<std::size_t, 3> coordinate_bases{2, 3, 5};
inline constexpr std::array<std::size_t, 3> velocity_radius_bases{7, 11, 13};
inline constexpr std::array<std::size_t, 3> velocity_phase_bases{17, 19, 23};

consteval bool bases_are_disjoint(const auto& lhs, const auto& rhs)
{
  for (const auto a : lhs)
    for (const auto b : rhs)
      if (a == b)
        return false;
  return true;
}

static_assert(
  bases_are_disjoint(coordinate_bases, velocity_radius_bases) &&
  bases_are_disjoint(coordinate_bases, velocity_phase_bases) &&
  bases_are_disjoint(velocity_radius_bases, velocity_phase_bases));

/// @brief Returns one component of a Halton point for the given index and base.
inline PetscReal van_der_corput(std::size_t i, std::size_t base)
{
  PetscReal q = 0.0;
  PetscReal bk = 1.0 / static_cast<PetscReal>(base);
  while (i > 0) {
    q += static_cast<PetscReal>(i % base) * bk;
    i /= base;
    bk /= static_cast<PetscReal>(base);
  }
  return q;
}

}  // namespace halton

#endif  // SRC_UTILS_HALTON_GENERATOR_H
