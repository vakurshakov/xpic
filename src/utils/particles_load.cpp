#include "particles_load.h"

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
  PetscReal phi = 2.0 * M_PI * random_01();

  return Vector3R{
    cyl.center[X] + r * std::cos(phi),
    cyl.center[Y] + r * std::sin(phi),
    cyl.center[Z] + cyl.height * (random_01() - 0.5),
  };
}

Vector3R CoordinateOnAnnulus::operator()()
{
  PetscReal r = std::sqrt(POW2(ann.inner_r) +  //
    (POW2(ann.outer_r) - POW2(ann.inner_r)) * random_01());

  PetscReal phi = 2.0 * M_PI * random_01();

  return Vector3R{
    ann.center[X] + r * std::cos(phi),
    ann.center[Y] + r * std::sin(phi),
    ann.center[Z] + ann.height * (random_01() - 0.5),
  };
}


Vector3R PreciseMomentum::operator()(const Vector3R& /* coordinate */)
{
  return value;
}

PetscReal temperature_momentum(PetscReal temperature, PetscReal mass)
{
  return std::sqrt(-2.0 * (temperature * mass / mec2) * std::log(random_01()));
}

Vector3R MaxwellianMomentum::operator()(const Vector3R& /* coordinate */)
{
  Vector3R result{
    params.px +
      std::sin(2.0 * M_PI * random_01()) *
        temperature_momentum(params.Tx, params.m),

    params.py +
      std::sin(2.0 * M_PI * random_01()) *
        temperature_momentum(params.Ty, params.m),

    params.pz +
      std::sin(2.0 * M_PI * random_01()) *
        temperature_momentum(params.Tz, params.m),
  };

  if (tov)
    result /= std::sqrt(params.m * params.m + result.squared());
  return result;
}

Vector3R MaxwellCosinePerturbation::operator()(const Vector3R& coordinate)
{
  static const PetscReal Lx = (box.max[X] - box.min[X]);
  static const PetscReal Ly = (box.max[Y] - box.min[Y]);
  static const PetscReal Lz = (box.max[Z] - box.min[Z]);

  Vector3R v_m{
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Tx, params.m),
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Ty, params.m),
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Tz, params.m),
  };

  v_m /= std::sqrt(params.m * params.m + v_m.squared());

  Vector3R v_0{
    a[X] * sqrt(params.Tx / (params.m * mec2)),
    a[Y] * sqrt(params.Ty / (params.m * mec2)),
    a[Z] * sqrt(params.Tz / (params.m * mec2)),
  };

  v_m[X] += v_0[X] * std::cos(2.0 * M_PI * m[X] * coordinate[X] / Lx);
  v_m[Y] += v_0[Y] * std::cos(2.0 * M_PI * m[Y] * coordinate[Y] / Ly);
  v_m[Z] += v_0[Z] * std::cos(2.0 * M_PI * m[Z] * coordinate[Z] / Lz);

  return v_m;
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
