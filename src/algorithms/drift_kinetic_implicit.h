#ifndef SRC_ALGORITHMS_DRIFT_KINETIC_ESIRKEPOV_H
#define SRC_ALGORITHMS_DRIFT_KINETIC_ESIRKEPOV_H

#include "implicit_esirkepov.h"

class DriftKineticEsirkepov : public ImplicitEsirkepov {
public:
  DriftKineticEsirkepov(Vector3R*** E_g, Vector3R*** B_g, Vector3R*** J_g, Vector3R*** gradB_g);

  void interpolate(Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p,
    const Vector3R& rn, const Vector3R& r0);

  using ImplicitEsirkepov::interpolate;
  using ImplicitEsirkepov::decompose;

private:
  Vector3R*** gradB_g;  // Сетка градиента магнитного поля
};

#endif  // SRC_ALGORITHMS_DRIFT_KINETIC_ESIRKEPOV_H
