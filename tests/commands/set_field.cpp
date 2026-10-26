#include "src/commands/set_field.h"

#include "src/utils/world.h"

static constexpr char help[] = "Testing Yee staggering of SetField setters\n.";

using Expected = std::function<PetscReal(PetscInt c, const PetscReal r[3])>;

/// @brief Checks every component against `expected`, evaluated at the Yee
/// position `(idx[a] + stagger.offset[c][a]) * Dx[a]` of this component.
static PetscErrorCode check(
  const char* name, Vec vec, const Stagger& stagger, const Expected& expected)
{
  PetscFunctionBeginUser;
  DM da;
  PetscCall(VecGetDM(vec, &da));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  Arr arr;
  PetscCall(DMDAVecGetArray(da, vec, &arr));

  for (PetscInt z = start[Z]; z < start[Z] + size[Z]; z++) {
    for (PetscInt y = start[Y]; y < start[Y] + size[Y]; y++) {
      for (PetscInt x = start[X]; x < start[X] + size[X]; x++) {
        const PetscInt idx[3] = {x, y, z};

        for (PetscInt c = 0; c < 3; c++) {
          PetscReal r[3];
          for (PetscInt a = 0; a < 3; a++)
            r[a] = (idx[a] + stagger.offset[c][a]) * Dx[a];

          const PetscReal e = expected(c, r);
          PetscCheck(std::abs(arr[z][y][x][c] - e) < 1e-12, PETSC_COMM_WORLD, PETSC_ERR_USER,
            "%s: wrong value at (%d %d %d)[%d], got %g, expected %g", name, x, y, z, c, arr[z][y][x][c], e);
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(da, vec, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  // dx != dy != dz to catch axis mix-ups
  World::set_geometry(6.0, 4.0, 3.0, 1.0, 1.0, 0.5, 0.25, 1.0, 1.0);

  World world;
  PetscCall(world.initialize());

  Vec vec;
  PetscCall(DMCreateGlobalVector(world.da, &vec));

  const PetscReal two_pi = 2.0 * M_PI;

  for (const Stagger& stagger : {yee_E, yee_B}) {
    const char* tag = (&stagger == &yee_E) ? "E" : "B";

    {
      PetscCall(VecSet(vec, 0));
      PetscCall(SetUniformField{Vector3R{1.0, 2.0, 3.0}}(vec, stagger));
      PetscCall(check(tag, vec, stagger, [](PetscInt c, const PetscReal*) {
        return c + 1.0;
      }));
    }

    {
      const Vector3R amp{1.0, 2.0, 3.0}, m{1.0, 2.0, 1.0}, phase{0.1, 0.2, 0.3};
      PetscCall(VecSet(vec, 0));
      PetscCall(SetCosineField{amp, m, phase}(vec, stagger));
      PetscCall(check(tag, vec, stagger, [&](PetscInt c, const PetscReal r[3]) {
        return amp[c] * std::cos(two_pi * m[c] / Geom[c] * r[c] + phase[c]);
      }));
    }

    {
      const Vector3R amp{1.0, 2.0, 3.0}, m{1.0, 2.0, 1.0};
      PetscCall(VecSet(vec, 0));
      PetscCall(SetGeneralCosineField{amp, m}(vec, stagger));
      PetscCall(check(tag, vec, stagger, [&](PetscInt c, const PetscReal r[3]) {
        PetscReal phase = 0;
        for (PetscInt a = 0; a < 3; a++)
          phase += two_pi * m[a] / Geom[a] * r[a];
        return amp[c] * std::cos(phase);
      }));
    }

    {
      const Vector3R value{1.0, 2.0, 3.0};
      const PetscInt axis = X;
      const PetscReal length = 10.0, origin = 0.5;
      PetscCall(SetGradientField{value, axis, length, origin}(vec, stagger));
      PetscCall(check(tag, vec, stagger, [&](PetscInt c, const PetscReal r[3]) {
        return value[c] * (1.0 + (r[axis] - origin) / length);
      }));
    }

    {
      const PetscReal value = 2.0, cx = 3.0, cy = 1.0, radius = 5.0;
      PetscCall(SetAzimuthalField{value, cx, cy, radius}(vec, stagger));
      PetscCall(check(tag, vec, stagger, [&](PetscInt c, const PetscReal r[3]) {
        if (c == X)
          return -value * (r[Y] - cy) / radius;
        if (c == Y)
          return +value * (r[X] - cx) / radius;
        return 0.0;
      }));
    }
  }

  PetscCall(VecDestroy(&vec));
  PetscCall(world.finalize());

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
