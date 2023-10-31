#ifndef SRC_UTILS_RANDOM_GENERATOR_H
#define SRC_UTILS_RANDOM_GENERATOR_H

#include <omp.h>
#include <random>

class Random_generator {
 public:

  static inline std::mt19937& get() {
    static Random_generator single_instance;
    return single_instance.gen;
  }

 private:
  std::random_device rd;
  std::mt19937 gen = std::mt19937(rd());

  Random_generator() = default;

  Random_generator(const Random_generator&) = delete;
  Random_generator& operator=(const Random_generator&) = delete;

  Random_generator(Random_generator&&) = delete;
  Random_generator& operator=(Random_generator&&) = delete;
};

inline double random_01() {
  static std::uniform_real_distribution distribution(0.0, 1.0);

  double value;

  #pragma omp critical
  value = distribution(Random_generator::get());

  return value;
}

inline int random_sign() {
  static std::bernoulli_distribution distribution(0.5);

  int value;

  #pragma omp critical
  value = distribution(Random_generator::get()) ? +1 : -1;

  return value;
}

#endif // SRC_UTILS_RANDOM_GENERATOR_H
