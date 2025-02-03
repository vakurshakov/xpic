#include "particles.h"

#include "src/algorithms/boris_push.h"
#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/impls/ecsimcorr/simulation.h"

namespace ecsimcorr {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world, parameters), simulation_(simulation)
{
  PetscFunctionBeginUser;
  DM da = world.da;
  PetscCallVoid(DMCreateLocalVector(da, &local_currI));
  PetscCallVoid(DMCreateLocalVector(da, &local_currJe));
  PetscCallVoid(DMCreateGlobalVector(da, &global_currI));
  PetscCallVoid(DMCreateGlobalVector(da, &global_currJe));

  PetscClassIdRegister("ecsimcorr::Particles", &classid);
  PetscLogEventRegister("first_push", classid, &events[0]);
  PetscLogEventRegister("ecsim_curr", classid, &events[1]);
  PetscLogEventRegister("second_push", classid, &events[2]);
  PetscLogEventRegister("final_update", classid, &events[3]);
  PetscFunctionReturnVoid();
}


PetscErrorCode Particles::first_push()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(world.da, local_currJe, &currJe));

  PetscLogEventBegin(events[0], local_currJe, 0, 0, 0);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage) {
    for (auto& point : cell) {
      const Vector3R old_r = point.r;

      BorisPush push;
      push.update_r((0.5 * dt), point, *this);

      Shape shape;
      shape.setup(old_r, point.r, shape_radius2, shape_func2);
      decompose_esirkepov_current(shape, point);
    }
  }

  PetscLogEventEnd(events[0], local_currJe, 0, 0, 0);

  PetscCall(DMDAVecRestoreArrayWrite(world.da, local_currJe, &currJe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @note `Particles::update_cells()` _must_ be called before this routine
/// @note `!assembled` state may be forced on particles sort by the other one
PetscErrorCode Particles::fill_ecsim_current(
  MatStencil* coo_i, MatStencil* coo_j, PetscReal* coo_v, bool assembled)
{
  PetscFunctionBeginUser;
  Vec local_B;
  Vector3R*** B;

  /// @note There is no ideological meaning to call `DMGlobalToLocal()`
  /// for each particles sort, they all will produce the same result.
  DM da = world.da;
  PetscCall(DMGetLocalVector(da, &local_B));
  PetscCall(DMGlobalToLocal(da, simulation_.B, INSERT_VALUES, local_B));

  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayWrite(da, local_currI, &currI));

  PetscLogEventBegin(events[1], local_B, local_currI, 0, 0);

  constexpr PetscInt shs = POW2(3 * POW3(3));

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (PetscInt g = 0; g < geom_nz * geom_ny * geom_nx; ++g) {
    const auto& cell = storage[g];
    if (cell.empty())
      continue;

    /// @note Calculating the proper offset of this thread into global array
    PetscInt off = 0;

    for (PetscInt omp_g = 0; omp_g < g; ++omp_g)
      off += shs * (PetscInt)!storage[omp_g].empty();

#if MAT_SET_VALUES_COO
    if (!assembled) {
      MatStencil* coo_ci = coo_i + off;
      MatStencil* coo_cj = coo_j + off;
      fill_matrix_indices(g, coo_ci, coo_cj);
    }
#endif
    PetscReal* coo_cv = coo_v + off;

    for (const auto& point : cell) {
      Shape shape;
      shape.setup(point.r, shape_radius1, shape_func1);

      Vector3R B_p;
      SimpleInterpolation interpolation(shape);
      interpolation.process({}, {{B_p, B}});

      decompose_ecsim_current(shape, point, B_p, coo_cv);
    }
  }

  matrix_indices_assembled = true;

  PetscLogEventEnd(events[1], local_B, local_currI, 0, 0);

  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_currI, &currI));
  PetscCall(DMRestoreLocalVector(da, &local_B));

  PetscCall(DMLocalToGlobal(da, local_currI, ADD_VALUES, global_currI));
  PetscCall(VecAXPY(simulation_.currI, 1.0, global_currI));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::second_push()
{
  PetscFunctionBeginUser;
  Vec local_E;
  Vec local_B;
  Vector3R*** E;
  Vector3R*** B;

  DM da = world.da;
  PetscCall(DMGetLocalVector(da, &local_E));
  PetscCall(DMGetLocalVector(da, &local_B));
  PetscCall(DMGlobalToLocal(da, simulation_.Ep, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, simulation_.B, INSERT_VALUES, local_B));

  PetscCall(DMDAVecGetArrayRead(da, local_E, &E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayWrite(da, local_currJe, &currJe));

  PetscLogEventBegin(events[2], 0, 0, 0, 0);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage) {
    for (auto& point : cell) {
      const Vector3R old_r = point.r;

      Shape shape;
      shape.setup(point.r, shape_radius1, shape_func1);

      Vector3R E_p;
      Vector3R B_p;
      SimpleInterpolation interpolation(shape);
      interpolation.process({{E_p, E}}, {{B_p, B}});

      BorisPush push;
      push.update_fields(E_p, B_p);
      push.update_vEB(dt, point, *this);
      push.update_r((0.5 * dt), point, *this);

      shape.setup(old_r, point.r, shape_radius2, shape_func2);
      decompose_esirkepov_current(shape, point);
    }
  }

  PetscLogEventEnd(events[2], 0, 0, 0, 0);

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_currJe, &currJe));

  PetscCall(DMLocalToGlobal(da, local_currJe, ADD_VALUES, global_currJe));
  PetscCall(VecAXPY(simulation_.currJe, 1.0, global_currJe));

  PetscCall(DMRestoreLocalVector(da, &local_E));
  PetscCall(DMRestoreLocalVector(da, &local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::final_update()
{
  PetscFunctionBeginUser;
  PetscCall(MatMultAdd(simulation_.matL, simulation_.Ep, global_currI, global_currI));
  PetscCall(VecDot(global_currI, simulation_.Ep, &pred_w));
  PetscCall(VecDot(global_currJe, simulation_.Ec, &corr_w));

  PetscReal K0 = energy;
  PetscLogEventBegin(events[3], 0, 0, 0, 0);

  calculate_energy();
  PetscReal K = energy;

  PetscReal lambda2 = 1.0 + dt * (corr_w - pred_w) / K;
  PetscReal lambda = std::sqrt(lambda2);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage)
    for (auto& point : cell)
      point.p *= lambda;

  PetscLogEventEnd(events[3], 0, 0, 0, 0);

  lambda_dK = (lambda2 - 1.0) * K;
  pred_dK = K - K0;
  corr_dK = lambda2 * K - K0;
  energy = lambda2 * K;

  LOG("  Velocity renormalization for \"{}\"", parameters.sort_name);
  LOG("    predicted field work [(ECSIM) * E_pred]: {:.7f}", pred_w);
  LOG("    corrected field work [(Esirkepov) * E_corr]: {:.7f}", corr_w);
  LOG("    lambda: {:.7f}, lambda^2: {:.7f}", lambda, lambda2);
  LOG("    d(energy) pred.: {:.7f}, corr.: {:.7f}, lambda: {:.7f}", pred_dK, corr_dK, lambda_dK);
  LOG("    energy prev.: {:.7f}, curr.: {:.7f}, diff: {:.7f}", K0, energy, energy - K0 /* == corr_dK */);

  PetscFunctionReturn(PETSC_SUCCESS);
}


void Particles::decompose_esirkepov_current(const Shape& shape, const Point& point)
{
  PetscFunctionBeginUser;
  const PetscReal alpha =
    charge(point) * density(point) / (particles_number(point) * (6.0 * dt));

  EsirkepovDecomposition decomposition(shape, alpha);
  PetscCallVoid(decomposition.process(currJe));
}


/// @note Also decomposes `Simulation::matL`
#if !MAT_SET_VALUES_COO
void Particles::decompose_ecsim_current(const Shape& shape, const Point& point,
  const Vector3R& B_p, PetscReal* /* coo_v */)
#else
void Particles::decompose_ecsim_current(
  const Shape& shape, const Point& point, const Vector3R& B_p, PetscReal* coo_v)
#endif
{
  PetscFunctionBeginHot;
  const Vector3R& v = point.p;

  Vector3R b = 0.5 * dt * charge(point) / mass(point) * B_p;

  PetscReal betaI = density(point) * charge(point) /
    (particles_number(point) * (1.0 + b.squared()));

  PetscReal betaL = charge(point) / mass(point) * betaI;

  Vector3R I_p = betaI * (v + v.cross(b) + b * v.dot(b));

  SimpleDecomposition decomposition(shape, I_p);
  PetscCallVoid(decomposition.process(currI));

  constexpr PetscReal shape_tolerance = 1e-10;

#if !MAT_SET_VALUES_COO
  /// @todo Combine it with `Simple_decomposition::process()`?
  Mat matL = simulation_.matL;

  const PetscInt m = shape.size.elements_product();
  const PetscInt n = m;

  std::vector<MatStencil> idxm(m);
  std::vector<MatStencil> idxn(n);
  std::vector<PetscReal> values(static_cast<std::size_t>(m * n * POW2(3)), 0);

  /**
   * @brief indexing of `values` buffer for `MatSetValuesBlocked*()`
   * @param I block row, with `idxm[I]` being its index
   * @param J block column, with `idxn[J]` being its index
   * @param i row within a block, first component
   * @param j column within a block, second component
   */
  auto ind = [n](PetscInt I, PetscInt J, PetscInt i, PetscInt j) {
    return (I * 3 + i) * (3 * n) + (J * 3 + j);
  };

  // clang-format off
  for (PetscInt zi = 0; zi < shape.size[Z]; ++zi) {
  for (PetscInt yi = 0; yi < shape.size[Y]; ++yi) {
  for (PetscInt xi = 0; xi < shape.size[X]; ++xi) {
    PetscInt i = shape.s_p(zi, yi, xi);
    Vector3R si = shape.electric(i);

    idxm[i] = MatStencil{
      shape.start[Z] + zi,
      shape.start[Y] + yi,
      shape.start[X] + xi,
    };

    // Shifts of g'=g2 iteration
    for (PetscInt zj = 0; zj < shape.size[Z]; ++zj) {
    for (PetscInt yj = 0; yj < shape.size[Y]; ++yj) {
    for (PetscInt xj = 0; xj < shape.size[X]; ++xj) {
      PetscInt j = shape.s_p(zj, yj, xj);
      Vector3R sj = shape.electric(j);

      idxn[j] = MatStencil{
        shape.start[Z] + zj,
        shape.start[Y] + yj,
        shape.start[X] + xj,
      };

      if (si.abs_max() < shape_tolerance || sj.abs_max() < shape_tolerance)
        continue;

      values[ind(i, j, X, X)] = si[X] * sj[X] * betaL * (1.0   + b[X] * b[X]);
      values[ind(i, j, X, Y)] = si[X] * sj[Y] * betaL * (+b[Z] + b[X] * b[Y]);
      values[ind(i, j, X, Z)] = si[X] * sj[Z] * betaL * (-b[Y] + b[X] * b[Z]);

      values[ind(i, j, Y, Y)] = si[Y] * sj[Y] * betaL * (1.0   + b[Y] * b[Y]);
      values[ind(i, j, Y, X)] = si[Y] * sj[X] * betaL * (-b[Z] + b[Y] * b[X]);
      values[ind(i, j, Y, Z)] = si[Y] * sj[Z] * betaL * (+b[X] + b[Y] * b[Z]);

      values[ind(i, j, Z, Z)] = si[Z] * sj[Z] * betaL * (1.0   + b[Z] * b[Z]);
      values[ind(i, j, Z, X)] = si[Z] * sj[X] * betaL * (+b[Y] + b[X] * b[Z]);
      values[ind(i, j, Z, Y)] = si[Z] * sj[Y] * betaL * (-b[X] + b[Y] * b[Z]);
    }}}  // g'=g2
  }}}  // g=g1
  // clang-format on

  #pragma omp critical
  {
    // cannot use `PetscCall()`, omp section cannot be broken by return statement
    MatSetValuesBlockedStencil(
      matL, m, idxm.data(), n, idxn.data(), values.data(), ADD_VALUES);
  }
  PetscFunctionReturnVoid();
#else
  constexpr PetscInt shw = 3;

  /// @note It is an offset from particle `shape` indexing into `coo_v` one.
  const Vector3I off{
    shape.start[X] - ((PetscInt)point.x() - 1),
    shape.start[Y] - ((PetscInt)point.y() - 1),
    shape.start[Z] - ((PetscInt)point.z() - 1),
  };

  auto s_gg = [&](PetscInt g1, PetscInt g2) {
    Vector3I vg1{
      off[X] + g1 % shape.size[X],
      off[Y] + (g1 / shape.size[X]) % shape.size[Y],
      off[Z] + (g1 / shape.size[X]) / shape.size[Y],
    };

    Vector3I vg2{
      off[X] + g2 % shape.size[X],
      off[Y] + (g2 / shape.size[X]) % shape.size[Y],
      off[Z] + (g2 / shape.size[X]) / shape.size[Y],
    };

    return //
      ((vg1[Z] * shw + vg1[Y]) * shw + vg1[X]) * POW3(shw) +
      ((vg2[Z] * shw + vg2[Y]) * shw + vg2[X]);
  };

  const PetscReal matB[3][3]{
    {1.0 + b[X] * b[X], +b[Z] + b[X] * b[Y], -b[Y] + b[X] * b[Z]},
    {-b[Z] + b[Y] * b[X], 1.0 + b[Y] * b[Y], +b[X] + b[Y] * b[Z]},
    {+b[Y] + b[Z] * b[X], -b[X] + b[Z] * b[Y], 1.0 + b[Z] * b[Z]},
  };

  for (PetscInt g1 = 0; g1 < shape.size.elements_product(); ++g1) {
    for (PetscInt g2 = 0; g2 < shape.size.elements_product(); ++g2) {
      Vector3R s1 = shape.electric(g1);
      Vector3R s2 = shape.electric(g2);

      if (s1.abs_max() < shape_tolerance || s2.abs_max() < shape_tolerance)
        continue;

      PetscInt gg = s_gg(g1, g2);

      for (PetscInt c1 = 0; c1 < 3; ++c1)
        for (PetscInt c2 = 0; c2 < 3; ++c2)
          coo_v[ind(gg, c1, c2)] += s1[c1] * s2[c2] * betaL * matB[c1][c2];
    }
  }
  PetscFunctionReturnVoid();
#endif
}

void Particles::fill_matrix_indices(
  PetscInt g, MatStencil* coo_i, MatStencil* coo_j)
{
  PetscFunctionBeginUser;
  constexpr PetscInt shw = 3;

  Vector3I vg{
    g % geom_nx,
    (g / geom_nx) % geom_ny,
    (g / geom_nx) / geom_ny,
  };

  for (PetscInt g1 = 0; g1 < POW3(shw); ++g1) {
    for (PetscInt g2 = 0; g2 < POW3(shw); ++g2) {
      PetscInt gg = g1 * POW3(shw) + g2;

      Vector3I vg1{
        vg[X] + g1 % shw - 1,
        vg[Y] + (g1 / shw) % shw - 1,
        vg[Z] + (g1 / shw) / shw - 1,
      };

      Vector3I vg2{
        vg[X] + g2 % shw - 1,
        vg[Y] + (g2 / shw) % shw - 1,
        vg[Z] + (g2 / shw) / shw - 1,
      };

      for (PetscInt c = 0; c < 3; ++c) {
        coo_i[ind(gg, X, c)] = MatStencil{vg1[Z], vg1[Y], vg1[X], X};
        coo_i[ind(gg, Y, c)] = MatStencil{vg1[Z], vg1[Y], vg1[X], Y};
        coo_i[ind(gg, Z, c)] = MatStencil{vg1[Z], vg1[Y], vg1[X], Z};

        coo_j[ind(gg, c, X)] = MatStencil{vg2[Z], vg2[Y], vg2[X], X};
        coo_j[ind(gg, c, Y)] = MatStencil{vg2[Z], vg2[Y], vg2[X], Y};
        coo_j[ind(gg, c, Z)] = MatStencil{vg2[Z], vg2[Y], vg2[X], Z};
      }
    }
  }
  PetscFunctionReturnVoid();
}


/// @note Ideally, since we use one global Lapenta matrix, test on
/// `matrix_indices_assembled` should include particles of all sorts.
PetscErrorCode Particles::update_cells()
{
  PetscFunctionBeginUser;
  for (PetscInt g = 0; g < geom_nz * geom_ny * geom_nx; ++g) {
    auto it = storage[g].begin();
    while (it != storage[g].end()) {
      auto ng = indexing::s_g(  //
        static_cast<PetscInt>(it->z() / dz),  //
        static_cast<PetscInt>(it->y() / dy),  //
        static_cast<PetscInt>(it->x() / dx));

      if (ng == g) {
        it = std::next(it);
        continue;
      }

      if (matrix_indices_assembled && storage[ng].empty())
        LOG("  Indices assembly is broken by \"{}\"", parameters.sort_name);

      matrix_indices_assembled &= !storage[ng].empty();

      storage[ng].emplace_back(*it);
      it = storage[g].erase(it);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


Particles::~Particles()
{
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&local_currI));
  PetscCallVoid(VecDestroy(&local_currJe));
  PetscCallVoid(VecDestroy(&global_currI));
  PetscCallVoid(VecDestroy(&global_currJe));
  PetscFunctionReturnVoid();
}

PetscErrorCode Particles::calculate_energy()
{
  PetscFunctionBeginUser;
  energy = 0.0;

#pragma omp parallel for reduction(+ : energy), \
  schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage) {
    for (auto& point : cell) {
      energy += 0.5 * (mass(point) / particles_number(point)) * point.p.squared();
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(local_currI, 0.0));
  PetscCall(VecSet(local_currJe, 0.0));
  PetscCall(VecSet(global_currI, 0.0));
  PetscCall(VecSet(global_currJe, 0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}


}  // namespace ecsimcorr
