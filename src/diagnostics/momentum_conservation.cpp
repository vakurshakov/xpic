#include "momentum_conservation.h"

#include "src/algorithms/boris_push.h"
#include "src/utils/configuration.h"
#include "src/utils/shape.h"


MomentumConservation::MomentumConservation(
  DM da, Vec E, Vec B, std::vector<const interfaces::Particles*> particles)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/momentum_conservation.txt"),
    da(da),
    E(E),
    B(B),
    particles(particles)
{
}

PetscErrorCode MomentumConservation::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&En0));
  PetscCall(VecDestroy(&Bn0));
  PetscCall(TableDiagnostic::finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MomentumConservation::initialize()
{
  PetscFunctionBeginUser;
  std::fill_n(std::back_inserter(P0), particles.size(), 0);
  std::fill_n(std::back_inserter(P1), particles.size(), 0);
  std::fill_n(std::back_inserter(QE), particles.size(), 0);
  std::fill_n(std::back_inserter(QVB), particles.size(), 0);

  PetscCall(VecDuplicate(E, &En0));
  PetscCall(VecDuplicate(B, &Bn0));
  PetscCall(VecCopy(E, En0));
  PetscCall(VecCopy(B, Bn0));

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
    auto&& qvb = QVB[i];

    add(13, "Px_" + name, "{: .6e}", p1[X]);
    add(13, "Py_" + name, "{: .6e}", p1[Y]);
    add(13, "Pz_" + name, "{: .6e}", p1[Z]);
    add(13, "QEx_" + name, "{: .6e}", qe[X]);
    add(13, "QEy_" + name, "{: .6e}", qe[Y]);
    add(13, "QEz_" + name, "{: .6e}", qe[Z]);
    add(13, "QVBx_" + name, "{: .6e}", qvb[X]);
    add(13, "QVBy_" + name, "{: .6e}", qvb[Y]);
    add(13, "QVBz_" + name, "{: .6e}", qvb[Z]);

    err = (p1 - p0) / dt - qe - qvb;
    sum += err;
    freq = 0;

    if (auto denom = (p1 + p0).length(); std::abs(denom) > PETSC_SMALL)
      freq = ((p1 - p0).length() / denom) / (0.5 * dt);

    add(13, "N2dP_" + name, "{: .6e}", err.length());
    add(13, "nuP_" + name, "{: .6e}", freq);

    p0 = p1;
  }

  add(13, "N2dP", "{: .6e}", sum.length());

  PetscCall(VecCopy(E, En0));
  PetscCall(VecCopy(B, Bn0));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @details Balance `(P^{n+1} - P^n) / dt = QE + QVB` is exact for the time centering
 * of `basic` scheme: the push uses `E^n` and `B^{n+1/2} = (B^n + B^{n+1}) / 2`
 * (both half-steps of `B` use `rot E^n`), interpolated at the half-step
 * position `r^{n+1/2} = r^{n+1} - v^{n+1} * dt / 2`. Boris step reads
 * `v^{n+1} - v^n = a + b x v`, where `a = (q/m) dt E_p`, `b = -(q/m) dt B_p` and
 * `v = (v^n + v^{n+1}) / 2`, so `v` is recovered from `v^{n+1}` alone by
 * solving `(1 + b x / 2) v = v^{n+1} - a / 2`.
 */
PetscErrorCode MomentumConservation::calculate()
{
  PetscFunctionBeginUser;
  Vec Bnh, En0_loc, Bnh_loc;
  PetscCall(DMGetGlobalVector(da, &Bnh));
  PetscCall(VecCopy(Bn0, Bnh));
  PetscCall(VecAXPBY(Bnh, 0.5, 0.5, B));
  PetscCall(DMGetLocalVector(da, &En0_loc));
  PetscCall(DMGetLocalVector(da, &Bnh_loc));
  PetscCall(DMGlobalToLocal(da, En0, INSERT_VALUES, En0_loc));
  PetscCall(DMGlobalToLocal(da, Bnh, INSERT_VALUES, Bnh_loc));

  const PetscReal**** Ea;
  const PetscReal**** Ba;
  PetscCall(DMDAVecGetArrayDOFRead(da, En0_loc, &Ea));
  PetscCall(DMDAVecGetArrayDOFRead(da, Bnh_loc, &Ba));

  Shape shape;

  for (PetscInt i = 0; i < (PetscInt)particles.size(); ++i) {
    auto& sort = particles[i];
    const PetscReal Np = (PetscReal)sort->parameters.Np;
    const PetscReal m = sort->parameters.m / Np;
    const PetscReal q = sort->parameters.q / Np;
    const PetscReal qm = sort->parameters.q / sort->parameters.m;

    auto& px = P1[i][X] = 0;
    auto& py = P1[i][Y] = 0;
    auto& pz = P1[i][Z] = 0;
    auto& qex = QE[i][X] = 0;
    auto& qey = QE[i][Y] = 0;
    auto& qez = QE[i][Z] = 0;
    auto& qvbx = QVB[i][X] = 0;
    auto& qvby = QVB[i][Y] = 0;
    auto& qvbz = QVB[i][Z] = 0;

#pragma omp parallel for private(shape), \
  reduction(+ : px, py, pz, qex, qey, qez, qvbx, qvby, qvbz)
    for (auto&& cell : sort->storage) {
      for (auto&& point : cell) {
        px += m * point.px();
        py += m * point.py();
        pz += m * point.pz();

        shape.setup(BorisPush::updated_r(-0.5 * dt, point));

        PetscInt i, gx, gy, gz;
        Vector3R E_p, B_p;

        for (i = 0; i < shape.size.elements_product(); ++i) {
          gx = shape.start[X] + i % shape.size[X];
          gy = shape.start[Y] + (i / shape.size[X]) % shape.size[Y];
          gz = shape.start[Z] + (i / shape.size[X]) / shape.size[Y];

          const auto Es = shape.electric(i);
          const auto Bs = shape.magnetic(i);

          for (PetscInt c = 0; c < 3; ++c) {
            E_p[c] += Ea[gz][gy][gx][c] * Es[c];
            B_p[c] += Ba[gz][gy][gx][c] * Bs[c];
          }
        }

        const BorisPush push(qm, E_p, B_p);
        const Vector3R vh = push.mean_vEB(dt, point.p);
        const Vector3R FE = q * E_p;
        const Vector3R FB = q * vh.cross(B_p);

        qex += FE[X];
        qey += FE[Y];
        qez += FE[Z];
        qvbx += FB[X];
        qvby += FB[Y];
        qvbz += FB[Z];
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(da, En0_loc, &Ea));
  PetscCall(DMDAVecRestoreArrayDOFRead(da, Bnh_loc, &Ba));
  PetscCall(DMRestoreLocalVector(da, &En0_loc));
  PetscCall(DMRestoreLocalVector(da, &Bnh_loc));
  PetscCall(DMRestoreGlobalVector(da, &Bnh));

  // Each rank sums only its own particles, `P1`, `QE` and `QVB` are laid out as 3 reals per sort
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, P1.data(), 3 * (PetscMPIInt)P1.size(), MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, QE.data(), 3 * (PetscMPIInt)QE.size(), MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, QVB.data(), 3 * (PetscMPIInt)QVB.size(), MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}
