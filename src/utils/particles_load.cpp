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
  PetscReal r = cyl.radius * sqrt(random_01());
  PetscReal phi = 2.0 * M_PI * random_01();

  return Vector3R{
    cyl.center[X] + r * cos(phi),
    cyl.center[Y] + r * sin(phi),
    cyl.center[Z] + cyl.height * (random_01() - 0.5),
  };
}

Vector3R CoordinateInBoxCosineHump::operator()()
{
  PetscReal center = 0.5 * (box.min[axis] + box.max[axis]);
  PetscReal half_length = 0.5 * (box.max[axis] - box.min[axis]);

  while (true) {
    Vector3R coordinate{
      box.min[X] + random_01() * (box.max[X] - box.min[X]),
      box.min[Y] + random_01() * (box.max[Y] - box.min[Y]),
      box.min[Z] + random_01() * (box.max[Z] - box.min[Z]),
    };

    PetscReal normalized = (coordinate[axis] - center) / half_length;
    PetscReal weight = 0.5 * (1.0 + cos(M_PI * normalized));

    if (random_01() < weight)
      return coordinate;
  }
}

Vector3R CoordinateOnAnnulus::operator()()
{
  PetscReal r = sqrt(POW2(ann.inner_r) +  //
    (POW2(ann.outer_r) - POW2(ann.inner_r)) * random_01());

  PetscReal phi = 2.0 * M_PI * random_01();

  return Vector3R{
    ann.center[X] + r * cos(phi),
    ann.center[Y] + r * sin(phi),
    ann.center[Z] + ann.height * (random_01() - 0.5),
  };
}

Vector3R PreciseMomentum::operator()(const Vector3R& /* coordinate */)
{
  return value;
}

Vector3R thermal_momentum(const SortParameters& params)
{
  Vector3R T = {params.Tx, params.Ty, params.Tz};
  Vector3R result;

  for (Axis a : {X, Y, Z}) {
    PetscReal rand_vr = random_01();
    PetscReal rand_va = random_01();

    PetscReal sigma_p = sqrt(T[a] * params.m / mec2);
    result[a] = sigma_p * sqrt(-2.0 * log(rand_vr)) * sin(2.0 * M_PI * rand_va);
  }
  return result;
}

Vector3R thermal_velocity(const SortParameters& params)
{
  Vector3R velocity = thermal_momentum(params);
  velocity /= sqrt(params.m * params.m + velocity.squared());
  return velocity;
}

Vector3R MaxwellianMomentum::operator()(const Vector3R& /* coordinate */)
{
  Vector3R momentum = thermal_momentum(params);
  momentum[X] += params.px;
  momentum[Y] += params.py;
  momentum[Z] += params.pz;
  momentum /= sqrt(params.m * params.m + momentum.squared());
  return momentum;
}

Vector3R maxwellian_shifted_perturbation(  //
  const SortParameters& params, const BoxGeometry& box,
  const Vector3R& velocity, const Vector3R& wave_number, const Vector3R& phase,
  const Vector3R& coordinate, PetscReal (*trig_func)(PetscReal))
{
  PetscReal k[3]{
    2.0 * M_PI * wave_number[X] / (box.max[X] - box.min[X]),
    2.0 * M_PI * wave_number[Y] / (box.max[Y] - box.min[Y]),
    2.0 * M_PI * wave_number[Z] / (box.max[Z] - box.min[Z]),
  };

  Vector3R v_m = thermal_velocity(params);
  v_m[X] += velocity[X] * trig_func(k[X] * coordinate[X] + phase[X]);
  v_m[Y] += velocity[Y] * trig_func(k[Y] * coordinate[Y] + phase[Y]);
  v_m[Z] += velocity[Z] * trig_func(k[Z] * coordinate[Z] + phase[Z]);
  return v_m;
}

Vector3R MaxwellShiftedCosine::operator()(const Vector3R& coordinate)
{
  return maxwellian_shifted_perturbation(
    params, box, velocity, wave_number, phase, coordinate, cos);
}

Vector3R MaxwellShiftedSine::operator()(const Vector3R& coordinate)
{
  return maxwellian_shifted_perturbation(
    params, box, velocity, wave_number, phase, coordinate, sin);
}

Vector3R AngularMomentum::operator()(const Vector3R& coordinate)
{
  PetscReal x = coordinate[X] - center[X];
  PetscReal y = coordinate[Y] - center[Y];
  PetscReal r = hypot(x, y);

  Vector3R momentum = thermal_momentum(params);

  if (isinf(1.0 / r))
    return Vector3R{0, 0, params.pz} + momentum;

  return Vector3R{
    -params.px * (y / r) + momentum[X],
    +params.py * (x / r) + momentum[Y],
    +params.pz + momentum[Z],
  };
}
