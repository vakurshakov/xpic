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


SetGradientField::SetGradientField(
  const Vector3R& value, PetscInt axis, PetscReal length, PetscReal origin)
  : value_(value), axis_(axis), length_(length), origin_(origin)
{
}


SetAzimuthalField::SetAzimuthalField(
  PetscReal value, PetscReal center_x, PetscReal center_y, PetscReal radius)
  : value_(value), center_x_(center_x), center_y_(center_y), radius_(radius)
{
}

PetscErrorCode SetAzimuthalField::operator()(Vec vec)
{
  PetscFunctionBeginUser;
  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Vector3R*** arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));

  const bool linear = radius_ > 0.0;

#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    // Yee staggering: Bx lives at (i, j+1/2, k+1/2), By at (i+1/2, j, k+1/2),
    // so each transverse component is sampled at its own half-shifted node.
    const PetscReal rx_bx = x * dx - center_x_;
    const PetscReal ry_bx = (y + 0.5) * dy - center_y_;
    const PetscReal rx_by = (x + 0.5) * dx - center_x_;
    const PetscReal ry_by = y * dy - center_y_;
    // Linear: divide by the constant `radius`; normalized: divide by local r.
    const PetscReal denom_bx = linear ? radius_ : std::hypot(rx_bx, ry_bx);
    const PetscReal denom_by = linear ? radius_ : std::hypot(rx_by, ry_by);

    arr[z][y][x][X] =
      (!linear && denom_bx < 1e-10) ? 0.0 : -value_ * ry_bx / denom_bx;
    arr[z][y][x][Y] =
      (!linear && denom_by < 1e-10) ? 0.0 : +value_ * rx_by / denom_by;
    arr[z][y][x][Z] = 0.0;
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, vec, &arr));
  LOG("  Azimuthal field is set, |B| {}, center ({} {}), radius {}",
    value_, center_x_, center_y_, radius_);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetGradientField::operator()(Vec vec)
{
  PetscFunctionBeginUser;
  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Vector3R*** arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));

  const PetscReal d[3] = {dx, dy, dz};

#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    const PetscInt idx[3] = {x, y, z};
    const PetscReal s = idx[axis_] * d[axis_];
    const PetscReal scale = 1.0 + (s - origin_) / length_;

    arr[z][y][x][X] = value_[X] * scale;
    arr[z][y][x][Y] = value_[Y] * scale;
    arr[z][y][x][Z] = value_[Z] * scale;
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, vec, &arr));
  LOG("  Gradient field is set, value ({} {} {}), axis {}, length {}, origin {}",
    REP3_A(value_), axis_, length_, origin_);
  PetscFunctionReturn(PETSC_SUCCESS);
}


SetCosineField::SetCosineField(
  BoxGeometry field_box, const Vector3R& field_amplitude,
  const Vector3R& field_wave_number)
  : box(std::move(field_box)),
    amplitude(field_amplitude),
    wave_number(field_wave_number)
{
}

PetscErrorCode SetCosineField::operator()(Vec vec)
{
  PetscFunctionBeginUser;
  const Vector3R box_size = box.max - box.min;

  if (box_size[X] <= 0.0 || box_size[Y] <= 0.0 || box_size[Z] <= 0.0) {
    throw std::runtime_error(std::format(
      "SetCosineField requires positive box sizes, got ({}, {}, {})",
      box_size[X], box_size[Y], box_size[Z]));
  }

  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Vector3R*** arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));

  const PetscReal two_pi = 2.0 * M_PI;
  const PetscReal kx = two_pi * wave_number[X] / box_size[X];
  const PetscReal ky = two_pi * wave_number[Y] / box_size[Y];
  const PetscReal kz = two_pi * wave_number[Z] / box_size[Z];

#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    const PetscReal sx = x * dx;
    const PetscReal sy = y * dy;
    const PetscReal sz = z * dz;

    if (box.min[X] <= sx && sx < box.max[X]) {
      arr[z][y][x][X] +=
        amplitude[X] * std::cos(kx * (sx - box.min[X]));
    }

    if (box.min[Y] <= sy && sy < box.max[Y]) {
      arr[z][y][x][Y] +=
        amplitude[Y] * std::cos(ky * (sy - box.min[Y]));
    }

    if (box.min[Z] <= sz && sz < box.max[Z]) {
      arr[z][y][x][Z] +=
        amplitude[Z] * std::cos(kz * (sz - box.min[Z]));
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, vec, &arr));

  LOG("  Cosine field perturbation is set, amplitude ({} {} {}), wave_number ({} {} {})",
    REP3_A(amplitude), REP3_A(wave_number));
  PetscFunctionReturn(PETSC_SUCCESS);
}

SetGeneralCosineField::SetGeneralCosineField(
  BoxGeometry field_box, const Vector3R& field_amplitude,
  const Vector3R& field_wave_number)
  : box(std::move(field_box)),
    amplitude(field_amplitude),
    wave_number(field_wave_number)
{
}

PetscErrorCode SetGeneralCosineField::operator()(Vec vec)
{
  PetscFunctionBeginUser;
  const Vector3R box_size = box.max - box.min;

  if (box_size[X] <= 0.0 || box_size[Y] <= 0.0 || box_size[Z] <= 0.0) {
    throw std::runtime_error(std::format(
      "SetGeneralCosineField requires positive box sizes, got ({}, {}, {})",
      box_size[X], box_size[Y], box_size[Z]));
  }

  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Vector3R*** arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));

  const PetscReal two_pi = 2.0 * M_PI;
  const PetscReal kx = two_pi * wave_number[X] / box_size[X];
  const PetscReal ky = two_pi * wave_number[Y] / box_size[Y];
  const PetscReal kz = two_pi * wave_number[Z] / box_size[Z];

#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    const PetscReal sx = x * dx;
    const PetscReal sy = y * dy;
    const PetscReal sz = z * dz;

    if (box.min[X] <= sx && sx < box.max[X] &&
        box.min[Y] <= sy && sy < box.max[Y] &&
        box.min[Z] <= sz && sz < box.max[Z]) {

      PetscReal phase = kx * (sx - box.min[X]) +
                        ky * (sy - box.min[Y]) +
                        kz * (sz - box.min[Z]);

      PetscReal cos_val = std::cos(phase);

      arr[z][y][x][X] += amplitude[X] * cos_val;
      arr[z][y][x][Y] += amplitude[Y] * cos_val;
      arr[z][y][x][Z] += amplitude[Z] * cos_val;
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, vec, &arr));

  LOG("  General cosine field perturbation is set, amplitude ({} {} {}), wave_number ({} {} {})",
    REP3_A(amplitude), REP3_A(wave_number));
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


PetscErrorCode SetApproximateMirrorField::operator()(Vec vec)
{
  PetscFunctionBeginUser;
  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Vector3R*** arr;
  PetscCall(DMDAVecGetArrayWrite(da, vec, &arr));

  PetscReal sz, sm;

#pragma omp parallel for private(sz, sm)
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    PetscInt x = start[X] + g % size[X];
    PetscInt y = start[Y] + (g / size[X]) % size[Y];
    PetscInt z = start[Z] + (g / size[X]) / size[Y];

    sz = (z + 0.5) * dz;
    sm = 1.5 * (x * dx - 0.5 * geom_x);
    arr[z][y][x][X] += get_B0(sz, +1) * sm * get_B1(sz, +1);
    arr[z][y][x][X] += get_B0(sz, -1) * sm * get_B1(sz, -1);

    sz = (z + 0.5) * dz;
    sm = 1.5 * (y * dy - 0.5 * geom_y);
    arr[z][y][x][X] += get_B0(sz, +1) * sm * get_B1(sz, +1);
    arr[z][y][x][X] += get_B0(sz, -1) * sm * get_B1(sz, -1);

    sz = z * dz;
    arr[z][y][x][Z] += get_B0(sz, +1);
    arr[z][y][x][Z] += get_B0(sz, -1);
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, vec, &arr));

  LOG("  Approximate magnetic mirror field is set!");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal SetApproximateMirrorField::get_B0(PetscReal z, PetscReal sign)
{
  return 0.5 * I * POW2(R) / std::pow(POW2(R) + POW2(z + 0.5 * sign * D), 1.5);
}

PetscReal SetApproximateMirrorField::get_B1(PetscReal z, PetscReal sign)
{
  return (z + 0.5 * sign * D) / (POW2(R) + POW2(z + 0.5 * sign * D));
}
