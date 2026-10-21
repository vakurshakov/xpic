#include "src/utils/operators.h"

#include "petscdm.h"
#include "src/utils/world.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing finite-difference operators: Yee stencils and vector identities\n.";

constexpr PetscReal tol = 1e-16;

using Arr4 = const PetscReal****;
using Expected = std::function<PetscReal(Arr4, REP4(PetscInt))>;

static PetscReal diff(Arr4 f, bool negative, PetscInt axis,  //
  PetscInt x, PetscInt y, PetscInt z, PetscInt c);

static PetscErrorCode check_structure(const char* name,  //
  Mat op, DM source_da, DM result_da, const Expected& expected);

static PetscErrorCode check_zero_composition(const char* name,  //
  Mat first, Mat second, DM source_da, DM middle_da, DM result_da);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  // dx != dy != dz to catch axis mix-ups
  World::set_geometry(8.0, 5.0, 4.0, 1.0, 1.0, 0.5, 0.25, 1.0, 1.0);

  World world;
  for (DMBoundaryType& bound : world.bounds) {
    bound = DM_BOUNDARY_PERIODIC;
  }
  PetscCall(world.initialize());

  PetscCall(DMSetMatrixPreallocateOnly(world.da, PETSC_FALSE));
  PetscCall(DMSetMatrixPreallocateSkip(world.da, PETSC_TRUE));

  // Index is `negative`
  Mat rot[2], grad[2], div[2];
  PetscCall(Rotor(world.da).create_positive(&rot[0]));
  PetscCall(Rotor(world.da).create_negative(&rot[1]));
  PetscCall(Gradient(world.da).create_positive(&grad[0]));
  PetscCall(Gradient(world.da).create_negative(&grad[1]));
  PetscCall(Divergence(world.da).create_positive(&div[0]));
  PetscCall(Divergence(world.da).create_negative(&div[1]));

  for (bool negative : {false, true}) {
    const std::string shift = negative ? "negative" : "positive";

    // (rot f)_c = D_{c+1} f_{c+2} - D_{c+2} f_{c+1}
    PetscCall(check_structure(("rot " + shift).c_str(), rot[negative], world.da, world.da,
      [&](Arr4 f, PetscInt x, PetscInt y, PetscInt z, PetscInt c)
      {
            const PetscInt a = (c + 1) % 3;
            const PetscInt b = (c + 2) % 3;
            return                                //
              diff(f, negative, a, x, y, z, b) -  //
              diff(f, negative, b, x, y, z, a);
      }));

    PetscCall(check_structure(("grad " + shift).c_str(), grad[negative], world.da_rho, world.da,
      [&](Arr4 f, PetscInt x, PetscInt y, PetscInt z, PetscInt c)
      {
            return diff(f, negative, c, x, y, z, 0);
      }));

    PetscCall(check_structure(("div " + shift).c_str(), div[negative], world.da, world.da_rho,
      [&](Arr4 f, PetscInt x, PetscInt y, PetscInt z, PetscInt /* c */)
      {
            return                                //
              diff(f, negative, X, x, y, z, X) +  //
              diff(f, negative, Y, x, y, z, Y) +  //
              diff(f, negative, Z, x, y, z, Z);
      }));
  }

  // Mixed terms cancel only if both operators use the same shift
  for (bool negative : {false, true}) {
    const std::string name = negative ? "negative" : "positive";

    PetscCall(check_zero_composition(("div(rot) " + name).c_str(),  //
      rot[negative], div[negative], world.da, world.da, world.da_rho));

    PetscCall(check_zero_composition(("rot(grad) " + name).c_str(),  //
      grad[negative], rot[negative], world.da_rho, world.da, world.da));
  }

  for (bool negative : {false, true}) {
    PetscCall(MatDestroy(&grad[negative]));
    PetscCall(MatDestroy(&rot[negative]));
    PetscCall(MatDestroy(&div[negative]));
  }

  PetscCall(world.finalize());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Finite difference of component `c` along `axis` with the Yee shift.
static PetscReal diff(Arr4 f, bool negative, PetscInt axis,  //
  PetscInt x, PetscInt y, PetscInt z, PetscInt c)
{
  PetscInt lo[3]{x, y, z};
  PetscInt hi[3]{x, y, z};
  negative ? lo[axis]-- : hi[axis]++;
  return (f[hi[Z]][hi[Y]][hi[X]][c] - f[lo[Z]][lo[Y]][lo[X]][c]) / Dx[axis];
}

/// @brief Compares `op * random` with the stencil, written explicitly with ghost values.
static PetscErrorCode check_structure(const char* name,  //
  Mat op, DM source_da, DM result_da, const Expected& expected)
{
  PetscFunctionBeginUser;
  Vec source, result, local;
  PetscCall(DMGetGlobalVector(source_da, &source));
  PetscCall(DMGetGlobalVector(result_da, &result));
  PetscCall(VecSetRandom(source, nullptr));
  PetscCall(MatMult(op, source, result));

  PetscCall(DMGetLocalVector(source_da, &local));
  PetscCall(DMGlobalToLocal(source_da, source, INSERT_VALUES, local));

  PetscInt dof;
  Vector3I start, size;
  PetscCall(DMDAGetInfo(result_da, nullptr, REP3(nullptr), REP3(nullptr), &dof, nullptr, REP4(nullptr)));
  PetscCall(DMDAGetCorners(result_da, REP3_A(&start), REP3_A(&size)));

  Arr4 src, res;
  PetscCall(DMDAVecGetArrayDOFRead(source_da, local, &src));
  PetscCall(DMDAVecGetArrayDOFRead(result_da, result, &res));

  for (PetscInt z = start[Z]; z < start[Z] + size[Z]; z++) {
    for (PetscInt y = start[Y]; y < start[Y] + size[Y]; y++) {
      for (PetscInt x = start[X]; x < start[X] + size[X]; x++) {
        for (PetscInt c = 0; c < dof; c++) {
          const PetscReal e = expected(src, x, y, z, c);

          PetscCheck(equal_tol(res[z][y][x][c], e, tol), PETSC_COMM_WORLD, PETSC_ERR_USER,
            "%s: wrong value at (%d %d %d)[%d], got %g, expected %g", name, x, y, z, c, res[z][y][x][c], e);
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(result_da, result, &res));
  PetscCall(DMDAVecRestoreArrayDOFRead(source_da, local, &src));
  PetscCall(DMRestoreGlobalVector(source_da, &source));
  PetscCall(DMRestoreGlobalVector(result_da, &result));
  PetscCall(DMRestoreLocalVector(source_da, &local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Checks `second * first * random = 0`, `first` maps `source_da` to `middle_da`.
static PetscErrorCode check_zero_composition(const char* name,  //
  Mat first, Mat second, DM source_da, DM middle_da, DM result_da)
{
  PetscFunctionBeginUser;
  Vec source, middle, result;
  PetscCall(DMGetGlobalVector(source_da, &source));
  PetscCall(DMGetGlobalVector(middle_da, &middle));
  PetscCall(DMGetGlobalVector(result_da, &result));
  PetscCall(VecSetRandom(source, nullptr));
  PetscCall(MatMult(first, source, middle));
  PetscCall(MatMult(second, middle, result));

  PetscReal norm_middle, norm_result;
  PetscCall(VecNorm(middle, NORM_INFINITY, &norm_middle));
  PetscCall(VecNorm(result, NORM_INFINITY, &norm_result));

  PetscCheck(norm_middle > 1 && norm_result < tol, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "%s: |intermediate| = %g (should be non-trivial), |result| = %g (should be zero)", name, norm_middle, norm_result);

  PetscCall(DMRestoreGlobalVector(source_da, &source));
  PetscCall(DMRestoreGlobalVector(middle_da, &middle));
  PetscCall(DMRestoreGlobalVector(result_da, &result));
  PetscFunctionReturn(PETSC_SUCCESS);
}
