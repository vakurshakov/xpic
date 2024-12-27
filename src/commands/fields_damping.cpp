#include "fields_damping.h"

#include "src/utils/configuration.h"

class FieldsDamping::DampForCircle {
public:
  DampForCircle(const CircleGeometry& geom, PetscReal coefficient);
  void operator()(
    PetscInt z, PetscInt y, PetscInt x, Vector3R*** f, PetscReal& energy);
  CircleGeometry geom_;
  PetscReal coefficient_;
};


FieldsDamping::FieldsDamping(DM da, const std::vector<Vec>& storages,
  const CircleGeometry& geom, PetscReal coefficient)
  : da_(da), storages_(storages), damp_(DampForCircle(geom, coefficient))
{
}

PetscErrorCode FieldsDamping::execute(timestep_t /* t */)
{
  PetscFunctionBeginUser;
  damped_energy_ = 0.0;

  Vector3I start;
  Vector3I end;
  PetscCall(DMDAGetCorners(da_, REP3_A(&start), REP3_A(&end)));
  end += start;

  for (Vec storage : storages_) {
    Vector3R*** f;
    PetscCall(DMDAVecGetArrayWrite(da_, storage, reinterpret_cast<void*>(&f)));

    // clang-format off
    for (PetscInt z = start.z(); z < end.z(); ++z) {
    for (PetscInt y = start.y(); y < end.y(); ++y) {
    for (PetscInt x = start.x(); x < end.x(); ++x) {
      damp_(z, y, x, f, damped_energy_);
    }}}
    // clang-format on
    PetscCall(DMDAVecRestoreArrayWrite(da_, storage, reinterpret_cast<void*>(&f)));
  }

  LOG("  Fields are damped, additional energy runoff is {}", damped_energy_);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal FieldsDamping::get_damped_energy() const
{
  return damped_energy_;
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

  if (delta < delta0) {
    damping = 1.0 - coefficient_ * POW2(delta / width - 1.0);
  }

  Vector3R& ff = f[z][y][x];
  energy += 0.5 * ff.squared() * (dx * dy * dz) * (1.0 - POW2(damping));
  ff *= damping;
}
