#include "momentum_conservation.h"

#include "src/utils/configuration.h"
#include "src/utils/shape.h"


MomentumConservation::MomentumConservation(
  DM da, Vec E, std::vector<const interfaces::Particles*> particles)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/momentum_conservation.txt"),
    da(da),
    E(E),
    particles(particles)
{
  std::fill_n(std::back_inserter(P0), particles.size(), 0.0);
  std::fill_n(std::back_inserter(P1), particles.size(), 0.0);
  std::fill_n(std::back_inserter(QE), particles.size(), 0.0);
}

PetscErrorCode MomentumConservation::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(calculate());

  for (PetscInt i = 0; i < (PetscInt)particles.size(); ++i) {
    P0[i] = P1[i];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MomentumConservation::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(calculate());

  add(6, "Time", "{:d}", t);

  Vector3R err, sum;
  PetscReal freq;

  for (PetscInt i = 0; i < (PetscInt)particles.size(); ++i) {
    auto&& name = particles[i]->parameters.sort_name;
    auto&& p0 = P0[i];
    auto&& p1 = P1[i];
    auto&& qe = QE[i];

    add(13, "Px_" + name, "{: .6e}", p1[X]);
    add(13, "Py_" + name, "{: .6e}", p1[Y]);
    add(13, "Pz_" + name, "{: .6e}", p1[Z]);
    add(13, "QEx_" + name, "{: .6e}", qe[X]);
    add(13, "QEy_" + name, "{: .6e}", qe[Y]);
    add(13, "QEz_" + name, "{: .6e}", qe[Z]);

    err = (p1 - p0) / dt - qe;
    sum += err;
    freq = 0;

    if (auto denom = (p1 + p0).length(); abs(denom) > PETSC_SMALL)
      freq = ((p1 - p0).length() / denom) / (0.5 * dt);

    add(13, "N2dP_" + name, "{: .6e}", err.length());
    add(13, "fP_" + name, "{: .6e}", freq);

    p0 = p1;
  }

  add(13, "N2dP", "{: .6e}", sum.length());
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode MomentumConservation::calculate()
{
  PetscFunctionBeginUser;
  Vec El;
  PetscCall(DMGetLocalVector(da, &El));
  PetscCall(DMGlobalToLocal(da, E, INSERT_VALUES, El));

  const PetscReal**** Ea;
  PetscCall(DMDAVecGetArrayDOFRead(da, El, &Ea));

  Shape shape;

  for (PetscInt i = 0; i < (PetscInt)particles.size(); ++i) {
    auto& sort = particles[i];
    PetscReal Np = (PetscReal)sort->parameters.Np;
    PetscReal m = sort->parameters.m / Np;
    PetscReal q = sort->parameters.q / Np;

    auto& px = P1[i][X] = 0;
    auto& py = P1[i][Y] = 0;
    auto& pz = P1[i][Z] = 0;
    auto& qex = QE[i][X] = 0;
    auto& qey = QE[i][Y] = 0;
    auto& qez = QE[i][Z] = 0;

#pragma omp parallel for private(shape), \
  reduction(+ : px, py, pz, qex, qey, qez)
    for (auto&& cell : sort->storage) {
      for (auto&& point : cell) {
        shape.setup(point.r);

        PetscInt i, gx, gy, gz;

        for (i = 0; i < shape.size.elements_product(); ++i) {
          gx = shape.start[X] + i % shape.size[X];
          gy = shape.start[Y] + (i / shape.size[X]) % shape.size[Y];
          gz = shape.start[Z] + (i / shape.size[X]) / shape.size[Y];

          auto ns = shape(i, No, Z) * shape(i, No, Y) * shape(i, No, X);
          auto Es = shape.electric(i);

          px += m * point.px() * ns;
          py += m * point.py() * ns;
          pz += m * point.pz() * ns;
          qex += q * Ea[gz][gy][gx][X] * Es[X];
          qey += q * Ea[gz][gy][gx][Y] * Es[Y];
          qez += q * Ea[gz][gy][gx][Z] * Es[Z];
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(da, El, &Ea));
  PetscCall(DMRestoreLocalVector(da, &El));
  PetscFunctionReturn(PETSC_SUCCESS);
}
