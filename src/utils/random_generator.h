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

  std::random_device rd;
  std::mt19937 gen = std::mt19937(rd());
};

inline double random_01()
{
  static std::uniform_real_distribution distribution(0.0, 1.0);

  double value;

#pragma omp critical
  value = distribution(RandomGenerator::get());

  return value;
}

inline PetscInt random_sign()
{
  static std::bernoulli_distribution distribution(0.5);

  PetscInt value;

#pragma omp critical
  value = distribution(RandomGenerator::get()) ? +1 : -1;

  return value;
}

#endif  // SRC_UTILS_RANDOM_GENERATOR_H
