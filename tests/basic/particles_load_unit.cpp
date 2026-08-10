#include "src/utils/particles_load.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
  if (!condition)
    throw std::runtime_error(message);
}

void test_antithetic_temperature()
{
  SortParameters params{};
  params.m = 1.0;
  params.Tx = 20.0;
  params.Ty = 10.0;
  params.Tz = 0.1;

  MaxwellianVelocityQuiet loader(params);
  constexpr std::size_t number_of_pairs = 65'536;
  PetscReal second_moment[3]{};

  for (std::size_t pair = 0; pair < number_of_pairs; ++pair) {
    const Vector3R first = loader(Vector3R{});
    const Vector3R second = loader(Vector3R{});

    require((first + second).abs_max() == 0.0,
      "MaxwellianVelocityQuiet pair is not exactly antithetic");
    require(first.squared() < 1.0 && second.squared() < 1.0,
      "MaxwellianVelocityQuiet emitted a superluminal velocity");

    for (Axis axis : {X, Y, Z})
      second_moment[axis] += first[axis] * first[axis];
  }

  const PetscReal temperatures[3] = {params.Tx, params.Ty, params.Tz};
  for (Axis axis : {X, Y, Z}) {
    const PetscReal measured =
      second_moment[axis] / static_cast<PetscReal>(number_of_pairs);
    const PetscReal expected = temperatures[axis] / (params.m * mec2);
    const PetscReal relative_error = std::abs(measured / expected - 1.0);
    require(relative_error < 0.005,
      "MaxwellianVelocityQuiet temperature error exceeds 0.5%");
  }
}

void test_subluminal_rejection()
{
  SortParameters params{};
  params.m = 1.0;
  params.Tx = mec2;
  params.Ty = mec2;
  params.Tz = mec2;

  MaxwellianVelocityQuiet loader(params);
  for (std::size_t pair = 0; pair < 128; ++pair) {
    const Vector3R first = loader(Vector3R{});
    const Vector3R second = loader(Vector3R{});
    require(first.squared() < 1.0 && second.squared() < 1.0,
      "MaxwellianVelocityQuiet rejection did not enforce |v| < 1");
    require((first + second).abs_max() == 0.0,
      "MaxwellianVelocityQuiet rejection broke antithetic pairing");
  }
}

}  // namespace

int main()
{
  try {
    test_antithetic_temperature();
    test_subluminal_rejection();
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
