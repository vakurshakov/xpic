#include "set_field.h"

#include <petscdmda.h>

#include "src/utils/configuration.h"

SetField::SetField(Vec field, Vec field0, Setter&& setup)
  : field(field), field0(field0), setup(std::move(setup))
{
}

PetscErrorCode SetField::execute(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  PetscCall(setup(field));
  if (field0)
    PetscCall(VecAXPY(field0, 1.0, field));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode SetUniformField::operator()(Vec vec)
{
  PetscFunctionBeginUser;
  PetscCall(VecStrideSet(vec, X, value[X]));
  PetscCall(VecStrideSet(vec, Y, value[Y]));
  PetscCall(VecStrideSet(vec, Z, value[Z]));
  LOG("  Uniform field is set, value ({} {} {})", REP3_A(value));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode SetAzimuthalField::operator()(Vec vec)
{
  PetscFunctionBeginUser;
  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Arr arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));

  const bool linear = radius > 0.0;

#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    // Yee staggering: Bx lives at (i, j+1/2, k+1/2), By at (i+1/2, j, k+1/2),
    // so each transverse component is sampled at its own half-shifted node.
    const PetscReal rx_bx = x * dx - center_x;
    const PetscReal ry_bx = (y + 0.5) * dy - center_y;
    const PetscReal rx_by = (x + 0.5) * dx - center_x;
    const PetscReal ry_by = y * dy - center_y;
    // Linear: divide by the constant `radius`; normalized: divide by local r.
    const PetscReal denom_bx = linear ? radius : std::hypot(rx_bx, ry_bx);
    const PetscReal denom_by = linear ? radius : std::hypot(rx_by, ry_by);

    arr[z][y][x][X] =
      (!linear && denom_bx < 1e-10) ? 0.0 : -value * ry_bx / denom_bx;
    arr[z][y][x][Y] =
      (!linear && denom_by < 1e-10) ? 0.0 : +value * rx_by / denom_by;
    arr[z][y][x][Z] = 0.0;
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, vec, &arr));
  LOG("  Azimuthal field is set, value {}, center ({} {}), radius {}", value, center_x, center_y, radius);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetGradientField::operator()(Vec vec)
{
  PetscFunctionBeginUser;
  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Arr arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));

  const PetscReal d[3] = {dx, dy, dz};

#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    const PetscInt idx[3] = {x, y, z};
    const PetscReal s = idx[axis] * d[axis];
    const PetscReal scale = 1.0 + (s - origin) / length;

    arr[z][y][x][X] = value[X] * scale;
    arr[z][y][x][Y] = value[Y] * scale;
    arr[z][y][x][Z] = value[Z] * scale;
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, vec, &arr));
  LOG("  Gradient field is set, value ({} {} {}), axis {}, length {}, origin {}", REP3_A(value), axis, length, origin);
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode SetCosineField::operator()(Vec vec)
{
  PetscFunctionBeginUser;
  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Arr arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));

  const PetscReal two_pi = 2.0 * M_PI;
  const PetscReal kx = two_pi * wave_number[X] / geom_x;
  const PetscReal ky = two_pi * wave_number[Y] / geom_y;
  const PetscReal kz = two_pi * wave_number[Z] / geom_z;

#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    arr[z][y][x][X] += amplitude[X] * std::cos(kx * x * dx + phase[X]);
    arr[z][y][x][Y] += amplitude[Y] * std::cos(ky * y * dy + phase[Y]);
    arr[z][y][x][Z] += amplitude[Z] * std::cos(kz * z * dz + phase[Z]);
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, vec, &arr));

  LOG("  Cosine field perturbation is set, amplitude ({} {} {}), wave_number ({} {} {}), phase ({} {} {})", REP3_A(amplitude), REP3_A(wave_number), REP3_A(phase));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetGeneralCosineField::operator()(Vec vec)
{
  PetscFunctionBeginUser;
  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Arr arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));

  const PetscReal two_pi = 2.0 * M_PI;
  const PetscReal kx = two_pi * wave_number[X] / geom_x;
  const PetscReal ky = two_pi * wave_number[Y] / geom_y;
  const PetscReal kz = two_pi * wave_number[Z] / geom_z;

#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    const PetscReal cos_val = std::cos(kx * x * dx + ky * y * dy + kz * z * dz);

    arr[z][y][x][X] += amplitude[X] * cos_val;
    arr[z][y][x][Y] += amplitude[Y] * cos_val;
    arr[z][y][x][Z] += amplitude[Z] * cos_val;
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, vec, &arr));

  LOG("  General cosine field perturbation is set, amplitude ({} {} {}), wave_number ({} {} {})", REP3_A(amplitude), REP3_A(wave_number));
  PetscFunctionReturn(PETSC_SUCCESS);
}


SetCoilsField::SetCoilsField(std::vector<Coil>&& coils)
  : coils(std::move(coils))
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

  Arr arr;
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

  LOG("  Coils field is set!");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal SetCoilsField::get_Br(PetscReal z, PetscReal r)
{
  PetscReal Br = 0.0;
  for (const auto& coil : coils) {
    PetscReal zc = z - coil.z0;
    Br += coil.I * coil.R * zc * get_integ_r(zc, r, coil.R);
  }
  return Br;
}

PetscReal SetCoilsField::get_Bz(PetscReal z, PetscReal r)
{
  PetscReal Bz = 0.0;
  for (const auto& coil : coils) {
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
