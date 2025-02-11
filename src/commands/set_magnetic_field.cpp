#include "set_magnetic_field.h"

#include "src/utils/configuration.h"

SetMagneticField::SetMagneticField(Vec storage, Setter&& setup)
  : storage_(storage), setup_(std::move(setup))
{
}

PetscErrorCode SetMagneticField::execute(timestep_t /* t */)
{
  PetscFunctionBeginUser;
  PetscCall(setup_(storage_));
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
  PetscReal* arr;
  PetscCall(VecGetArrayWrite(vec, &arr));

  PetscReal center_x = 0.5 * geom_x;
  PetscReal center_y = 0.5 * geom_y;

  PetscReal sx, sy, sz, r;

#pragma omp parallel for private(sx, sy, sz, r)
  for (PetscInt g = 0; g < geom_nz * geom_ny * geom_nx; ++g) {
    PetscReal x = (g % geom_nx) * dx /*       */ - center_x;
    PetscReal y = ((g / geom_nx) % geom_ny) * dy - center_y;
    PetscReal z = ((g / geom_nx) / geom_ny) * dz;

    sx = x;
    sy = y + 0.5;
    sz = z + 0.5;
    r = std::hypot(sx, sy);
    if (r < denominator_tolerance)
      r = denominator_tolerance;
    arr[g * 3 + X] += get_Br(sz, r) * sx / r;

    sy = y;
    sx = x + 0.5;
    sz = z + 0.5;
    r = std::hypot(sx, sy);
    if (r < denominator_tolerance)
      r = denominator_tolerance;
    arr[g * 3 + Y] += get_Br(sz, r) * sy / r;

    sz = z;
    sx = x + 0.5;
    sy = y + 0.5;
    r = std::hypot(sx, sy);
    if (r < denominator_tolerance)
      r = denominator_tolerance;
    arr[g * 3 + Z] += get_Bz(sz, r);
  }

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
    Bz += coil.I * coil.R * hp * get_integ_z(zc, r, coil.R);
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
  return integral;
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
  return integral;
}
