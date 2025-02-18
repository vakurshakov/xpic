#include "particles_load.hpp"

#include "src/utils/random_generator.h"
#include "src/utils/utils.h"

Vector3R PreciseCoordinate::operator()()
{
  return dot;
}


Vector3R CoordinateInBox::operator()()
{
  return Vector3R{
    box.min[X] + random_01() * (box.max[X] - box.min[X]),
    box.min[Y] + random_01() * (box.max[Y] - box.min[Y]),
    box.min[Z] + random_01() * (box.max[Z] - box.min[Z]),
  };
}


Vector3R CoordinateInCylinder::operator()()
{
  PetscReal r = cyl.radius * std::sqrt(random_01());
  PetscReal phi = 2.0 * std::numbers::pi * random_01();

  return Vector3R{
    cyl.center[X] + r * std::cos(phi),
    cyl.center[Y] + r * std::sin(phi),
    cyl.center[Z] + cyl.height * (random_01() - 0.5),
  };
}


CoordinateOnAnnulus::CoordinateOnAnnulus(const Vector3R& center,
  PetscReal inner_r, PetscReal outer_r, PetscReal height)
  : center(center),
    inner_r2(POW2(inner_r)),
    outer_r2(POW2(outer_r)),
    height(height)
{
}

Vector3R CoordinateOnAnnulus::operator()()
{
  PetscReal r = std::sqrt(inner_r2 + (outer_r2 - inner_r2) * random_01());
  PetscReal phi = 2.0 * std::numbers::pi * random_01();

  return Vector3R{
    center[X] + r * cos(phi),
    center[Y] + r * sin(phi),
    center[Z] + height * (random_01() - 0.5),
  };
}


Vector3R PreciseMomentum::operator()(const Vector3R& /* coordinate */)
{
  return value;
}


PetscReal temperature_momentum(PetscReal temperature, PetscReal mass)
{
  static constexpr PetscReal mec2 = 511.0;  // KeV

  PetscReal p = 0.0;
  do {
    p = std::sqrt(-2.0 * (temperature * mass / mec2) * std::log(random_01()));
  }
  while (std::isinf(p));
  return p;
}


Vector3R MaxwellianMomentum::operator()(const Vector3R& /* coordinate */)
{
  Vector3R result{
    params.px +
      std::sin(2.0 * std::numbers::pi * random_01()) *
        temperature_momentum(params.Tx, params.m),

    params.py +
      std::sin(2.0 * std::numbers::pi * random_01()) *
        temperature_momentum(params.Ty, params.m),

    params.pz +
      std::sin(2.0 * std::numbers::pi * random_01()) *
        temperature_momentum(params.Tz, params.m),
  };

  if (tov)
    result /= std::sqrt(params.m * params.m + result.squared());
  return result;
}


Vector3R AngularMomentum::operator()(const Vector3R& coordinate)
{
  PetscReal x = coordinate.x() - center[X];
  PetscReal y = coordinate.y() - center[Y];
  PetscReal r = std::hypot(x, y);

  Vector3R temperature_moment{
    temperature_momentum(params.Tx, params.m),
    temperature_momentum(params.Ty, params.m),
    temperature_momentum(params.Tz, params.m),
  };

  if (std::isinf(1.0 / r))
    return Vector3R{0.0, 0.0, params.pz} + temperature_moment;

  return Vector3R{
    -params.px * (y / r) + temperature_moment[X],
    +params.py * (x / r) + temperature_moment[Y],
    +params.pz + temperature_moment[Z],
  };
}
