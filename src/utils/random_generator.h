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

  std::mt19937 gen{
#if RANDOM_SEED
    std::random_device()()
#endif
  };
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
