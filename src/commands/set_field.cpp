#include "set_field.h"

#include <petscdmda.h>

#include "src/utils/configuration.h"

SetField::SetField(Vec field, Vec field0, const Stagger& stagger, Setter&& setup)
  : field(field), field0(field0), stagger(stagger), setup(std::move(setup))
{
}

PetscErrorCode SetField::execute(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  PetscCall(setup(field, stagger));
  if (field0)
    PetscCall(VecAXPY(field0, 1.0, field));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode SetUniformField::operator()(Vec vec, const Stagger& /* stagger */)
{
  PetscFunctionBeginUser;
  PetscCall(VecStrideSet(vec, X, value[X]));
  PetscCall(VecStrideSet(vec, Y, value[Y]));
  PetscCall(VecStrideSet(vec, Z, value[Z]));
  LOG("  Uniform field is set, value ({} {} {})", REP3_A(value));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode SetAzimuthalField::operator()(Vec vec, const Stagger& stagger)
{
  PetscFunctionBeginUser;
  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Arr arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));

  const bool linear = radius > 0.0;
  const auto& off = stagger.offset;

#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    const PetscReal rx_x = (x + off[X][X]) * dx - center_x;
    const PetscReal rx_y = (x + off[Y][X]) * dx - center_x;
    const PetscReal ry_x = (y + off[X][Y]) * dy - center_y;
    const PetscReal ry_y = (y + off[Y][Y]) * dy - center_y;

    // Linear: divide by the constant `radius`; normalized: divide by local r.
    const PetscReal denom_x = linear ? radius : std::hypot(rx_x, ry_x);
    const PetscReal denom_y = linear ? radius : std::hypot(rx_y, ry_y);

    arr[z][y][x][X] = (!linear && denom_x < 1e-10) ? 0.0
                                                   : -value * ry_x / denom_x;
    arr[z][y][x][Y] = (!linear && denom_y < 1e-10) ? 0.0
                                                   : +value * rx_y / denom_y;
    arr[z][y][x][Z] = 0.0;
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, vec, &arr));
  LOG("  Azimuthal field is set, value {}, center ({} {}), radius {}", value, center_x, center_y, radius);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetGradientField::operator()(Vec vec, const Stagger& stagger)
{
  PetscFunctionBeginUser;
  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Arr arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));


#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    const PetscInt idx[3] = {x, y, z};

    for (PetscInt c = 0; c < 3; c++) {
      const PetscReal s = (idx[axis] + stagger.offset[c][axis]) * Dx[axis];
      arr[z][y][x][c] = value[c] * (1.0 + (s - origin) / length);
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, vec, &arr));
  LOG("  Gradient field is set, value ({} {} {}), axis {}, length {}, origin {}", REP3_A(value), axis, length, origin);
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode SetCosineField::operator()(Vec vec, const Stagger& stagger)
{
  PetscFunctionBeginUser;
  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Arr arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));

  const PetscReal two_pi = 2.0 * M_PI;
  const PetscReal k[3]{
    two_pi * wave_number[X] / geom_x,
    two_pi * wave_number[Y] / geom_y,
    two_pi * wave_number[Z] / geom_z,
  };

#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    const PetscInt idx[3] = {x, y, z};

    for (PetscInt c = 0; c < 3; c++) {
      const PetscReal s = (idx[c] + stagger.offset[c][c]) * Dx[c];
      arr[z][y][x][c] += amplitude[c] * std::cos(k[c] * s + phase[c]);
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, vec, &arr));

  LOG("  Cosine field perturbation is set, amplitude ({} {} {}), wave_number ({} {} {}), phase ({} {} {})", REP3_A(amplitude), REP3_A(wave_number), REP3_A(phase));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetGeneralCosineField::operator()(Vec vec, const Stagger& stagger)
{
  PetscFunctionBeginUser;
  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Arr arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));

  const PetscReal two_pi = 2.0 * M_PI;
  const PetscReal k[3]{
    two_pi * wave_number[X] / geom_x,
    two_pi * wave_number[Y] / geom_y,
    two_pi * wave_number[Z] / geom_z,
  };

#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    const PetscInt idx[3] = {x, y, z};

    for (PetscInt c = 0; c < 3; c++) {
      PetscReal phase = 0;
      for (PetscInt a = 0; a < 3; a++)
        phase += k[a] * (idx[a] + stagger.offset[c][a]) * Dx[a];

      arr[z][y][x][c] += amplitude[c] * std::cos(phase);
    }
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

PetscErrorCode SetCoilsField::operator()(Vec vec, const Stagger& stagger)
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
  const auto& off = stagger.offset;

  PetscReal sx, sy, sz, r;

#pragma omp parallel for private(sx, sy, sz, r)
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    auto position = [&](PetscInt c)
    {
      sx = (x + off[c][X]) * dx - center_x;
      sy = (y + off[c][Y]) * dy - center_y;
      sz = (z + off[c][Z]) * dz;
      r = std::hypot(sx, sy);
    };

    position(X);
    arr[z][y][x][X] += get_Br(sz, r) * sx / r;

    position(Y);
    arr[z][y][x][Y] += get_Br(sz, r) * sy / r;

    position(Z);
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
