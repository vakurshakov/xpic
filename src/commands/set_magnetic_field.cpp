#include "set_magnetic_field.h"

#include <petscdmda.h>

#include "src/utils/configuration.h"

SetMagneticField::SetMagneticField(Vec B0, Vec B, Setter&& setup)
  : B0_(B0), B_(B), setup_(std::move(setup))
{
}

PetscErrorCode SetMagneticField::execute(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  PetscCall(setup_(B0_));
  if (B_)
    PetscCall(VecAXPY(B_, 1.0, B0_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


SetUniformField::SetUniformField(const Vector3R& value)
  : value_(value)
{
}

PetscErrorCode SetUniformField::operator()(Vec vec)
{
  PetscFunctionBeginUser;
  PetscCall(VecStrideSet(vec, X, value_[X]));
  PetscCall(VecStrideSet(vec, Y, value_[Y]));
  PetscCall(VecStrideSet(vec, Z, value_[Z]));
  LOG("  Uniform magnetic field is set, value ({} {} {})", REP3_A(value_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


SetCoilsField::SetCoilsField(std::vector<Coil>&& coils)
  : coils_(std::move(coils))
{
  for (PetscInt i = 0; i < N; ++i)
    cos[i] = std::cos(i * hp);
}

PetscErrorCode SetCoilsField::operator()(Vec vec)
{
  PetscFunctionBeginUser;
  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Vector3R*** arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));

  PetscReal center_x = 0.5 * geom_x;
  PetscReal center_y = 0.5 * geom_y;

  PetscReal sx, sy, sz, r;

#pragma omp parallel for private(sx, sy, sz, r)
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    sx = x * dx /*   */ - center_x;
    sy = (y + 0.5) * dy - center_y;
    sz = (z + 0.5) * dz;
    r = std::hypot(sx, sy);
    arr[z][y][x][X] += get_Br(sz, r) * sx / r;

    sy = y * dy /*   */ - center_y;
    sx = (x + 0.5) * dx - center_x;
    sz = (z + 0.5) * dz;
    r = std::hypot(sx, sy);
    arr[z][y][x][Y] += get_Br(sz, r) * sy / r;

    sz = z * dz;
    sx = (x + 0.5) * dx - center_x;
    sy = (y + 0.5) * dy - center_y;
    r = std::hypot(sx, sy);
    arr[z][y][x][Z] += get_Bz(sz, r);
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, vec, &arr));

  LOG("  Coils magnetic field is set!");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal SetCoilsField::get_Br(PetscReal z, PetscReal r)
{
  PetscReal Br = 0.0;
  for (const auto& coil : coils_) {
    PetscReal zc = z - coil.z0;
    Br += coil.I * coil.R * zc * get_integ_r(zc, r, coil.R);
  }
  return Br;
}

PetscReal SetCoilsField::get_Bz(PetscReal z, PetscReal r)
{
  PetscReal Bz = 0.0;
  for (const auto& coil : coils_) {
    PetscReal zc = z - coil.z0;
    Bz += coil.I * coil.R * get_integ_z(zc, r, coil.R);
  }
  return Bz;
}

PetscReal SetCoilsField::get_integ_r(PetscReal z, PetscReal r, PetscReal R)
{
  PetscReal integral = 0.0;
  PetscReal denominator;

  for (PetscInt i = 0; i < N; ++i) {
    denominator = POW2(z) + POW2(R) + POW2(r) - 2.0 * R * r * cos[i];
    if (std::abs(denominator) < denominator_tolerance)
      denominator = denominator_tolerance;
    integral += (cos[i] / (denominator * std::sqrt(denominator)));
  }
  return hp * integral;
}

PetscReal SetCoilsField::get_integ_z(PetscReal z, PetscReal r, PetscReal R)
{
  PetscReal integral = 0.0;
  PetscReal denominator;

  for (PetscInt i = 0; i < N; ++i) {
    denominator = POW2(z) + POW2(R) + POW2(r) - 2.0 * R * r * cos[i];
    if (std::abs(denominator) < denominator_tolerance)
      denominator = denominator_tolerance;
    integral += ((R - r * cos[i]) / (denominator * std::sqrt(denominator)));
  }
  return hp * integral;
}
