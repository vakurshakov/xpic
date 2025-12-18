#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic Esirkepov interpolation: simple field interpolation.\n"
  "Tests interpolation of arbitrary E, B, and gradB fields on a particle.\n";

using namespace implicit_test_utils;

constexpr Vector3R E0(1.0, 1.0, 1.0);

void get_analytical_fields(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = E0;
  B_p = Vector3R(r.y(),r.z(),r.x());
  gradB_p = r/r.length();
}

void get_grid_fields(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = E0;
  B_p = Vector3R(r.y()+0.5*dy,r.z()+0.5*dz,r.x()+0.5*dx);
  gradB_p = r/r.length();
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  InterpCase test_in_one_cell_1{
    .r0 = {1.,1.,1.},
    .rn = {1.5,1.3,2.},
    .analytic_fn = get_analytical_fields,
    .grid_fn = get_grid_fields,
  };

  InterpCase test_in_one_cell_2{
    .r0 = {1.8,1.6,1.4},
    .rn = {1.5,1.3,2.},
    .analytic_fn = get_analytical_fields,
    .grid_fn = get_grid_fields,
  };

  InterpCase test_without_displace_1{
    .r0 = {1.,1.,1.},
    .rn = {1.,2.,2.},
    .analytic_fn = get_analytical_fields,
    .grid_fn = get_grid_fields,
  };

  InterpCase test_without_displace_2{
    .r0 = {1.,1.,1.},
    .rn = {2.,1.,2.},
    .analytic_fn = get_analytical_fields,
    .grid_fn = get_grid_fields,
  };

  InterpCase test_without_displace_3{
    .r0 = {1.,1.,1.},
    .rn = {2.,2.,1.},
    .analytic_fn = get_analytical_fields,
    .grid_fn = get_grid_fields,
  };

  PetscCall(interpolation_test(test_in_one_cell_1));
  PetscCall(interpolation_test(test_in_one_cell_2));
  PetscCall(interpolation_test(test_without_displace_1));
  PetscCall(interpolation_test(test_without_displace_2));
  PetscCall(interpolation_test(test_without_displace_3));

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}