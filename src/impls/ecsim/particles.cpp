#include "particles.h"

#include "src/algorithms/boris_push.h"
#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/impls/ecsim/simulation.h"

namespace ecsim {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world, parameters), simulation_(simulation)
{
  DM da = world.da;
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da, &local_currI));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &global_currI));
}

PetscErrorCode Particles::first_push()
{
  PetscFunctionBeginUser;
#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage) {
    for (auto& point : cell) {
      BorisPush::update_r(dt, point);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::fill_ecsim_current(PetscReal* coo_v)
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(world.da, local_currI, &currI));

  PetscInt prev_g = 0;
  PetscInt off = 0;

#pragma omp parallel for firstprivate(prev_g, off)
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    if (storage[g].empty())
      continue;

    simulation_.get_array_offset(prev_g, g, off);
    prev_g = g;

    PetscReal* coo_cv = coo_v + off;
    for (const auto& point : storage[g]) {
      decompose_ecsim_current(point, coo_cv);
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(world.da, local_currI, &currI));
  PetscCall(DMLocalToGlobal(world.da, local_currI, ADD_VALUES, global_currI));
  PetscCall(VecAXPY(simulation_.currI, 1.0, global_currI));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @note Also decomposes `Simulation::matL`
void Particles::decompose_ecsim_current(const Point& point, PetscReal* coo_v)
{
  PetscFunctionBeginUser;
  const auto& [r, v] = point;

  PetscReal q = parameters.q;
  PetscReal m = parameters.m;
  PetscReal mpw = parameters.n / (PetscReal)parameters.Np;

  /// @todo Create some weights calculator
  PetscInt ixn, iyn, izn, ixs, iys, izs, ox, oy, oz;
  PetscReal xn, yn, zn, xs, ys, zs;
  PetscReal wnx[2], wny[2], wnz[2], wsx[2], wsy[2], wsz[2];

  xn = r[X] / dx;
  yn = r[Y] / dy;
  zn = r[Z] / dz;
  xs = xn - 0.5;
  ys = yn - 0.5;
  zs = zn - 0.5;

  ixn = (PetscInt)floor(xn);
  iyn = (PetscInt)floor(yn);
  izn = (PetscInt)floor(zn);
  ixs = (PetscInt)floor(xs);
  iys = (PetscInt)floor(ys);
  izs = (PetscInt)floor(zs);
  ox = ixs - ixn + 1;
  oy = iys - iyn + 1;
  oz = izs - izn + 1;

  wnx[1] = (xn - ixn);
  wny[1] = (yn - iyn);
  wnz[1] = (zn - izn);
  wnx[0] = 1 - wnx[1];
  wny[0] = 1 - wny[1];
  wnz[0] = 1 - wnz[1];

  wsx[1] = (xs - ixs);
  wsy[1] = (ys - iys);
  wsz[1] = (zs - izs);
  wsx[0] = 1 - wsx[1];
  wsy[0] = 1 - wsy[1];
  wsz[0] = 1 - wsz[1];

  Vector3R b = interpolate_B_s1(B, r) * ((0.5 * dt) * q / m);
  Vector3R I_p = q * mpw / (1. + b.squared()) * (v + v.cross(b) + v.dot(b) * b);
  PetscReal A_p = 0.5 * dt * dt * mpw * q * q / m / (1 + b.squared());

  const PetscReal matB[3][3]{
    {1.0 + b[X] * b[X], +b[Z] + b[X] * b[Y], -b[Y] + b[X] * b[Z]},
    {-b[Z] + b[Y] * b[X], 1.0 + b[Y] * b[Y], +b[X] + b[Y] * b[Z]},
    {+b[Y] + b[Z] * b[X], -b[X] + b[Z] * b[Y], 1.0 + b[Z] * b[Z]},
  };

  PetscInt c1, i1, j1, k1;
  PetscInt c2, i2, j2, k2;
  PetscInt in, jn, kn, is, js, ks;
  PetscInt i[3], j[3], ind;
  PetscReal s1[3], s2[3];

  for (k1 = 0; k1 < 2; ++k1) {
    for (j1 = 0; j1 < 2; ++j1) {
      for (i1 = 0; i1 < 2; ++i1) {
        in = ixn + i1;
        jn = iyn + j1;
        kn = izn + k1;
        is = ixs + i1;
        js = iys + j1;
        ks = izs + k1;

        s1[X] = wnz[k1] * wny[j1] * wsx[i1];
        s1[Y] = wnz[k1] * wsy[j1] * wnx[i1];
        s1[Z] = wsz[k1] * wny[j1] * wnx[i1];

#pragma omp atomic update
        currI[kn][jn][is][X] += s1[X] * I_p[X];
#pragma omp atomic update
        currI[kn][js][in][Y] += s1[Y] * I_p[Y];
#pragma omp atomic update
        currI[ks][jn][in][Z] += s1[Z] * I_p[Z];


        i[X] = (k1 * 2 + j1) * 3 + (ox + i1);
        i[Y] = (k1 * 3 + (oy + j1)) * 2 + i1;
        i[Z] = ((oz + k1) * 2 + j1) * 2 + i1;

        for (k2 = 0; k2 < 2; ++k2) {
          for (j2 = 0; j2 < 2; ++j2) {
            for (i2 = 0; i2 < 2; ++i2) {
              s2[X] = wsx[i2] * wny[j2] * wnz[k2];
              s2[Y] = wnx[i2] * wsy[j2] * wnz[k2];
              s2[Z] = wnx[i2] * wny[j2] * wsz[k2];

              j[X] = (k2 * 2 + j2) * 3 + (ox + i2);
              j[Y] = (k2 * 3 + (oy + j2)) * 2 + i2;
              j[Z] = ((oz + k2) * 2 + j2) * 2 + i2;

              for (c1 = 0; c1 < 3; c1++) {
                for (c2 = 0; c2 < 3; c2++) {
                  ind = (c1 * 3 + c2) * POW2(12) + (i[c1] * 12 + j[c2]);
                  coo_v[ind] += s1[c1] * s2[c2] * A_p * matB[c1][c2];
                }
              }
            }  // G'x
          }  // G'y
        }  // G'z
      }  // Gx
    }  // Gy
  }  // Gz
  PetscFunctionReturnVoid();
}

PetscErrorCode Particles::second_push()
{
  PetscFunctionBeginUser;
  PetscReal q = parameters.q;
  PetscReal m = parameters.m;

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage) {
    for (auto& point : cell) {
      Vector3R E_p = interpolate_E_s1(E, point.r);
      Vector3R B_p = interpolate_B_s1(B, point.r);

      BorisPush push(q / m, E_p, B_p);
      push.update_vEB(dt, point);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(local_currI, 0.0));
  PetscCall(VecSet(global_currI, 0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&local_currI));
  PetscCall(VecDestroy(&global_currI));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace ecsim
