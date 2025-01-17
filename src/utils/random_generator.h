#ifndef SRC_UTILS_RANDOM_GENERATOR_H
#define SRC_UTILS_RANDOM_GENERATOR_H

#include <omp.h>

#include <random>

class RandomGenerator {
public:
  static std::mt19937& get()
  {
    static RandomGenerator single_instance;
    return single_instance.gen;
  }

private:
  DEFAULT_MOVABLE(RandomGenerator);

  RandomGenerator() = default;
  ~RandomGenerator() = default;

#if RANDOM_SEED
  std::mt19937 gen{std::random_device()()};
#else
  std::mt19937 gen;
#endif
};

inline PetscReal random_01()
{
  static std::uniform_real_distribution distribution(0.0, 1.0);

  PetscReal value;

#pragma omp critical
  value = distribution(RandomGenerator::get());

  return value;
}

#endif  // SRC_UTILS_RANDOM_GENERATOR_H
