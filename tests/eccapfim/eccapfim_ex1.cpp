#include "src/impls/eccapfim/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Test of the field solver of \"eccapfim\" implementation. A standing wave\n"
  "E_y = a * cos(kx) is launched in vacuum (no particles) along x and is   \n"
  "advanced for 5 cycles. The result is compared with the exact recurrence \n"
  "of the Crank-Nicolson scheme, B^{n+1} = B^{n} - dt * rot(E^{n+1/2}) and \n"
  "E^{n+1} = E^{n} + dt * rot(B^{n+1/2}), in the Yee-staggered points.     \n";

static constexpr PetscReal amplitude = 1.0;
static constexpr PetscReal tol = 1e-7;

void overwrite_config();

static PetscErrorCode check(Vec vec, const Stagger& stagger, PetscInt c,
  PetscReal expected, PetscReal (*f)(PetscReal));

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();
  Configuration::save(get_out_dir(__FILE__));

  eccapfim::Simulation simulation;
  PetscCall(simulation.initialize());
  PetscCall(simulation.calculate());

  // B_z = b * sin(k * x_B), E_y = a * cos(k * x_E). Each step rotates (a, b)
  // by the angle 2 * atan(dt * kappa / 2), where kappa is the discrete wave number.
  const PetscReal kappa = 2.0 / dx * std::sin(M_PI * dx / geom_x);
  const PetscReal phi = 2.0 * std::atan(0.5 * dt * kappa);
  const PetscReal a = amplitude * std::cos(geom_nt * phi);
  const PetscReal b = amplitude * std::sin(geom_nt * phi);

  PetscCall(check(simulation.E, yee_E, Y, a, std::cos));
  PetscCall(check(simulation.B, yee_B, Z, b, std::sin));

  PetscCall(simulation.finalize());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  dx = 0.5;
  geom_nx = 10;
  geom_x = geom_nx * dx;

  dt = 1.0;
  geom_nt = 5;
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "eccapfim"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {
      "Geometry",
      {
        {"x", geom_x},
        {"y", geom_x},
        {"z", geom_x},
        {"t", geom_t},
        {"dx", dx},
        {"dy", dx},
        {"dz", dx},
        {"dt", dt},
        {"diagnose_period", geom_t},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Presets",
      {{
        {"command", "SetField"},
        {"field", "E"},
        {
          "setter",
          {
            {"name", "SetGeneralCosineField"},
            {"amplitude", {0.0, amplitude, 0.0}},
            {"wave_number", {1.0, 0.0, 0.0}},
          },
        },
      }},
    },
  });
}

/// @brief Checks the component `c` of `vec` to be `expected * f(k * x)`, where
/// `x` is a position of this component in the Yee cell, all others are zero.
static PetscErrorCode check(Vec vec, const Stagger& stagger, PetscInt c,
  PetscReal expected, PetscReal (*f)(PetscReal))
{
  PetscFunctionBeginUser;
  DM da;
  PetscCall(VecGetDM(vec, &da));

  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

  const PetscReal k = 2.0 * M_PI / geom_x;

  const PetscReal**** arr;
  PetscCall(DMDAVecGetArrayDOFRead(da, vec, &arr));
  for (PetscInt z = zs; z < zs + zm; ++z) {
    for (PetscInt y = ys; y < ys + ym; ++y) {
      for (PetscInt x = xs; x < xs + xm; ++x) {
        for (PetscInt i = 0; i < 3; ++i) {
          const PetscReal x_c = (x + stagger.offset[i][X]) * dx;
          const PetscReal e = (i == c) ? expected * f(k * x_c) : 0.0;
          PetscCheck(equal_tol(arr[z][y][x][i], e, tol), PETSC_COMM_WORLD, PETSC_ERR_USER,
            "Wrong value at (%d %d %d)[%d], got %g, expected %g", x, y, z, i, arr[z][y][x][i], e);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOFRead(da, vec, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}
