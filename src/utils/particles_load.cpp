#include "particles_load.hpp"

#include "src/utils/random_generator.h"
#include "src/utils/utils.h"

CoordinateInBox::CoordinateInBox(const Vector3R& min, const Vector3R& max)
  : min_(min), max_(max)
{
}

Vector3R CoordinateInBox::operator()()
{
  return Vector3R{
    min_[X] + random_01() * (max_[X] - min_[X]),
    min_[Y] + random_01() * (max_[Y] - min_[Y]),
    min_[Z] + random_01() * (max_[Z] - min_[Z]),
  };
}


CoordinateInCircle::CoordinateInCircle(PetscReal radius, const Vector3R& center)
  : radius_(radius), center_(center)
{
}

Vector3R CoordinateInCircle::operator()()
{
  PetscReal r = radius_ * std::sqrt(random_01());
  PetscReal phi = 2.0 * std::numbers::pi * random_01();

  return Vector3R{
    center_[X] + r * std::cos(phi),
    center_[Y] + r * std::sin(phi),
    center_[Z] + 0.0,
  };
}


CoordinateInCylinder::CoordinateInCylinder(
  PetscReal radius, PetscReal width, const Vector3R& center)
  : gen_(radius, center), width_(width)
{
}

Vector3R CoordinateInCylinder::operator()()
{
  Vector3R coordinate = gen_();
  return coordinate + Vector3R{0.0, 0.0, -0.5 * width_ * random_01()};
}


CoordinateOnAnnulus::CoordinateOnAnnulus(
  PetscReal inner_r, PetscReal outer_r, const Vector3R& center)
  : inner_r2_(POW2(inner_r)), outer_r2_(POW2(outer_r)), center_(center)
{
}

Vector3R CoordinateOnAnnulus::operator()()
{
  PetscReal r = std::sqrt(inner_r2_ + (outer_r2_ - inner_r2_) * random_01());
  PetscReal phi = 2.0 * std::numbers::pi * random_01();

  return Vector3R{
    center_[X] + r * cos(phi),
    center_[Y] + r * sin(phi),
    center_[Z] + 0.0,
  };
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


MaxwellianMomentum::MaxwellianMomentum(const SortParameters& params)
  : params_(params)
{
}

Vector3R MaxwellianMomentum::operator()(const Vector3R& /* coordinate */)
{
  return Vector3R{
    params_.px +
      std::sin(2.0 * std::numbers::pi * random_01()) *
        temperature_momentum(params_.Tx, params_.m),

    params_.py +
      std::sin(2.0 * std::numbers::pi * random_01()) *
        temperature_momentum(params_.Ty, params_.m),

    params_.pz +
      std::sin(2.0 * std::numbers::pi * random_01()) *
        temperature_momentum(params_.Tz, params_.m),
  };
}


AngularMomentum::AngularMomentum(
  const SortParameters& params, const Vector3R& center)
  : params_(params), center_(center)
{
}

Vector3R AngularMomentum::operator()(const Vector3R& coordinate)
{
  PetscReal x = coordinate.x() - center_[X];
  PetscReal y = coordinate.y() - center_[Y];
  PetscReal r = std::sqrt(x * x + y * y);

  Vector3R temperature_moment{
    temperature_momentum(params_.Tx, params_.m),
    temperature_momentum(params_.Ty, params_.m),
    temperature_momentum(params_.Tz, params_.m),
  };

  if (std::isinf(1.0 / r))
    return Vector3R{0.0, 0.0, params_.pz} + temperature_moment;

  return Vector3R{
    -params_.px * (y / r) + temperature_moment[X],
    +params_.py * (x / r) + temperature_moment[Y],
    +params_.pz + temperature_moment[Z],
  };
}
