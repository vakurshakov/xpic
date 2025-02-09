#include "particles_load.hpp"

#include "src/utils/random_generator.h"
#include "src/utils/utils.h"

PreciseCoordinate::PreciseCoordinate(const DotGeometry& dot)
  : dot(dot)
{
}

Vector3R PreciseCoordinate::operator()()
{
  return dot.dot;
}


CoordinateInBox::CoordinateInBox(const BoxGeometry& box)
  : box(box)
{
}

Vector3R CoordinateInBox::operator()()
{
  return Vector3R{
    box.min[X] + random_01() * (box.max[X] - box.min[X]),
    box.min[Y] + random_01() * (box.max[Y] - box.min[Y]),
    box.min[Z] + random_01() * (box.max[Z] - box.min[Z]),
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
  PetscReal radius, PetscReal height, const Vector3R& center)
  : gen_(radius, center), height_(height)
{
}

Vector3R CoordinateInCylinder::operator()()
{
  Vector3R coordinate = gen_();
  coordinate[Z] += height_ * (random_01() - 0.5);
  return coordinate;
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


PreciseMomentum::PreciseMomentum(const Vector3R& value)
  : value_(value)
{
}

Vector3R PreciseMomentum::operator()(const Vector3R& /* coordinate */)
{
  return value_;
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


MaxwellianMomentum::MaxwellianMomentum(const SortParameters& params, bool tov)
  : params_(params), tov_(tov)
{
}

Vector3R MaxwellianMomentum::operator()(const Vector3R& /* coordinate */)
{
  Vector3R result{
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

  if (tov_)
    result /= sqrt(params_.m * params_.m + result.squared());
  return result;
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
  PetscReal r = std::hypot(x, y);

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
