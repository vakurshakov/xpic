#include "fields_damping.h"

#include "src/utils/configuration.h"

class FieldsDamping::DampForBox {
public:
  DampForBox(const BoxGeometry& geom, PetscReal coefficient);
  void operator()(
    PetscInt z, PetscInt y, PetscInt x, Vector3R*** f, PetscReal& energy);
  BoxGeometry geom_;
  PetscReal coefficient_;
};

class FieldsDamping::DampForCircle {
public:
  DampForCircle(const CircleGeometry& geom, PetscReal coefficient);
  void operator()(
    PetscInt z, PetscInt y, PetscInt x, Vector3R*** f, PetscReal& energy);
  CircleGeometry geom_;
  PetscReal coefficient_;
};


FieldsDamping::FieldsDamping( //
  DM da, Vec E, Vec B, Vec B0, const BoxGeometry& geom, PetscReal coefficient)
  : da_(da), E_(E), B_(B), B0_(B0), damp_(DampForBox(geom, coefficient))
{
}

FieldsDamping::FieldsDamping( //
  DM da, Vec E, Vec B, Vec B0, const CircleGeometry& geom, PetscReal coefficient)
  : da_(da), E_(E), B_(B), B0_(B0), damp_(DampForCircle(geom, coefficient))
{
}

PetscErrorCode FieldsDamping::execute(timestep_t /* t */)
{
  PetscFunctionBeginUser;
  damped_energy_ = 0.0;

  PetscCall(VecAXPY(B_, -1.0, B0_));
  PetscCall(damping_implementation(E_));
  PetscCall(damping_implementation(B_));
  PetscCall(VecAXPY(B_, +1.0, B0_));

  LOG("  Fields are damped, additional energy runoff: {}", damped_energy_);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldsDamping::damping_implementation(Vec f)
{
  PetscFunctionBeginUser;
  Vector3I start;
  Vector3I end;
  PetscCall(DMDAGetCorners(da_, REP3_A(&start), REP3_A(&end)));
  end += start;

  Vector3R*** arr;
  PetscCall(DMDAVecGetArrayWrite(da_, f, reinterpret_cast<void*>(&arr)));

  // clang-format off
  for (PetscInt z = start.z(); z < end.z(); ++z) {
  for (PetscInt y = start.y(); y < end.y(); ++y) {
  for (PetscInt x = start.x(); x < end.x(); ++x) {
    damp_(z, y, x, arr, damped_energy_);
  }}}
  // clang-format on

  PetscCall(DMDAVecRestoreArrayWrite(da_, f, reinterpret_cast<void*>(&arr)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal FieldsDamping::get_damped_energy() const
{
  return damped_energy_;
}


FieldsDamping::DampForBox::DampForBox(
  const BoxGeometry& geom, PetscReal coefficient)
  : geom_(geom), coefficient_(coefficient)
{
}

void FieldsDamping::DampForBox::operator()(
  PetscInt z, PetscInt y, PetscInt x, Vector3R*** f, PetscReal& energy)
{
  const Vector3R r{
    x * dx,
    y * dy,
    z * dz,
  };

  bool is_outside = //
    (geom_.min[X] > r[X] || r[X] > geom_.max[X]) ||
    (geom_.min[Y] > r[Y] || r[Y] > geom_.max[Y]) ||
    (geom_.min[Z] > r[Z] || r[Z] > geom_.max[Z]);

  if (!is_outside)
    return;

  for (PetscInt i = 0; i < 3; ++i) {
    PetscReal width = 0.0;
    PetscReal delta = 0.0;

    if (r[i] > geom_.max[i]) {
      width = Geom[i] - geom_.max[i];
      delta = r[i] - geom_.max[i];
    }
    else if (r[i] < geom_.min[i]) {
      width = geom_.min[i] - 0;
      delta = r[i] - 0;
    }
    else continue;

    PetscReal damping = 1.0 - coefficient_ * POW2(delta / width - 1.0);

    energy += 0.5 * f[z][y][x].squared() * (dx * dy * dz) * (1.0 - POW2(damping));
    f[z][y][x] *= damping;
  }
}


FieldsDamping::DampForCircle::DampForCircle(
  const CircleGeometry& geom, PetscReal coefficient)
  : geom_(geom), coefficient_(coefficient)
{
}

void FieldsDamping::DampForCircle::operator()(
  PetscInt z, PetscInt y, PetscInt x, Vector3R*** f, PetscReal& energy)
{
  PetscInt cx = x * dx - geom_.center[X];
  PetscInt cy = y * dy - geom_.center[Y];
  PetscReal r2 = (POW2(cx) + POW2(cy));

  if (r2 < POW2(geom_.radius))
    return;

  PetscReal width = geom_.center[X] - geom_.radius;
  PetscReal delta = std::sqrt(r2) - geom_.radius;
  PetscReal delta0 = width * (1.0 + 1.0 / sqrt(coefficient_));

  /// @note To avoid negative damping coefficients, we check against `delta0`.
  PetscReal damping = 0.0;

  if (delta < delta0)
    damping = 1.0 - coefficient_ * POW2(delta / width - 1.0);

  energy += 0.5 * f[z][y][x].squared() * (dx * dy * dz) * (1.0 - POW2(damping));
  f[z][y][x] *= damping;
}
