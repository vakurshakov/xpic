#ifndef SRC_UTILS_HALTON_GENERATOR_H
#define SRC_UTILS_HALTON_GENERATOR_H

#include "src/utils/vector3.h"

inline PetscReal van_der_corput(std::size_t index, std::size_t base)
{
  PetscReal value = 0;
  PetscReal factor = 1.0 / base;
  while (index > 0) {
    value += (index % base) * factor;
    index /= base;
    factor /= base;
  }
  return value;
}

struct HaltonGenerator {
  Vector3R operator()()
  {
    Vector3R point{
      van_der_corput(next, 2),
      van_der_corput(next, 3),
      van_der_corput(next, 5),
    };
    next++;
    return point;
  }

  std::size_t next = 1;  // Skip the origin.
};

#endif  // SRC_UTILS_HALTON_GENERATOR_H
