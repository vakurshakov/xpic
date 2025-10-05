#include "drift_kinetic_implicit.h"

#include "src/algorithms/simple_interpolation.h"

DriftKineticEsirkepov::DriftKineticEsirkepov(
  Vector3R*** E_g, Vector3R*** B_g, Vector3R*** J_g, Vector3R*** gradB_g)
  : ImplicitEsirkepov(E_g, B_g, J_g), gradB_g(gradB_g)
{
}

void DriftKineticEsirkepov::interpolate(
  Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p,
  const Vector3R& rn, const Vector3R& r0)
{
  E_p = {};
  B_p = {};
  gradB_p = {};

  Vector3R dummyB;
  ImplicitEsirkepov::interpolate(E_p, dummyB, rn, r0);

  ::Shape shape;
  shape.setup(rn);

  SimpleInterpolation interpolation(shape);
  SimpleInterpolation::Context e_fields;
  SimpleInterpolation::Context b_fields{{B_p, B_g}};
  if (gradB_g != nullptr) {
    b_fields.emplace_back(gradB_p, gradB_g);
  }
  interpolation.process(e_fields, b_fields);
}
