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

Particles::~Particles()
{
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&local_currI));
  PetscCallVoid(VecDestroy(&local_currJe));
  PetscCallVoid(VecDestroy(&global_currI));
  PetscCallVoid(VecDestroy(&global_currJe));
  PetscFunctionReturnVoid();
}

PetscErrorCode Particles::init()
{
  PetscFunctionBeginUser;
  energy = 0.0;

#pragma omp parallel for reduction(+ : energy), \
  schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_)
    energy += 0.5 * (mass(point) / particles_number(point)) * point.p.squared();

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


PetscErrorCode Particles::first_push()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(world.da, local_currJe, &currJe));

  PetscLogEventBegin(events[0], local_currJe, 0, 0, 0);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_) {
    const Vector3R old_r = point.r;

    BorisPush push;
    push.update_r((0.5 * dt), point, *this);

    Shape shape;
    shape.setup(old_r, point.r, shape_radius2, shape_func2);
    decompose_esirkepov_current(shape, point);
  }

  PetscLogEventEnd(events[0], local_currJe, 0, 0, 0);

  PetscCall(DMDAVecRestoreArrayWrite(world.da, local_currJe, &currJe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::fill_ecsim_current(
  MatStencil* coo_i, MatStencil* coo_j, PetscReal* coo_v)
{
  PetscFunctionBeginUser;
  Vec local_B;
  Vector3R*** B;

  DM da = world.da;
  PetscCall(DMGetLocalVector(da, &local_B));
  PetscCall(DMGlobalToLocal(da, simulation_.B, INSERT_VALUES, local_B));

  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayWrite(da, local_currI, &currI));

  PetscLogEventBegin(events[1], local_B, local_currI, 0, 0);

  // #pragma omp parallel
  for (PetscInt i = 0; i < (PetscInt)points_.size(); ++i) {
    auto& point = points_[i];

    Shape shape;
    shape.setup(point.r, shape_radius1, shape_func1);

    Vector3R B_p;
    SimpleInterpolation interpolation(shape);
    interpolation.process({}, {{B_p, B}});

    decompose_ecsim_current(shape, i, B_p, coo_i, coo_j, coo_v);

    correct_coordinates(point);
  }

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
  for (auto& point : points_) {
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

    correct_coordinates(point);
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
  PetscReal K = 0.0;
  PetscLogEventBegin(events[3], 0, 0, 0, 0);

#pragma omp parallel for reduction(+ : K), \
  schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_)
    K += 0.5 * (mass(point) / particles_number(point)) * point.p.squared();

  PetscReal lambda2 = 1.0 + dt * (corr_w - pred_w) / K;
  PetscReal lambda = std::sqrt(lambda2);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_)
    point.p *= lambda;

  PetscLogEventEnd(events[3], 0, 0, 0, 0);

  lambda_dK = (lambda2 - 1.0) * K;
  pred_dK = K - K0;
  corr_dK = lambda2 * K - K0;
  energy = lambda2 * K;

  LOG("  Velocity renormalization for \"{}\"", parameters_.sort_name);
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


// NOLINTBEGIN(readability-function-cognitive-complexity)

/// @note Also decomposes `Simulation::matL`
#if !MAT_SET_VALUES_COO
void Particles::decompose_ecsim_current(const Shape& shape, PetscInt i,
  const Vector3R& B_p, MatStencil* /* coo_i */, MatStencil* /* coo_j */,
  PetscReal* /* coo_v */)
#else
void Particles::decompose_ecsim_current(const Shape& shape, PetscInt i,
  const Vector3R& B_p, MatStencil* coo_i, MatStencil* coo_j, PetscReal* coo_v)
#endif
{
  PetscFunctionBeginHot;
  const Point& point = points_[i];
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
  auto t_g = [&](PetscInt zi, PetscInt yi, PetscInt xi,  //
               PetscInt zj, PetscInt yj, PetscInt xj) {
    return ((zi * 3 + yi) * 3 + xi) * POW3(3) + ((zj * 3 + yj) * 3 + xj);
  };

  auto ind = [&](PetscInt g, PetscInt c1, PetscInt c2) {
    constexpr PetscInt shape_size = POW2(3 * POW3(3));
    return i * shape_size + g * 9 + (c1 * 3 + c2);
  };

  const PetscReal matB[3][3]{
    {1.0 + b[X] * b[X], +b[Z] + b[X] * b[Y], -b[Y] + b[X] * b[Z]},
    {-b[Z] + b[Y] * b[X], 1.0 + b[Y] * b[Y], +b[X] + b[Y] * b[Z]},
    {+b[Y] + b[Z] * b[X], -b[X] + b[Z] * b[Y], 1.0 + b[Z] * b[Z]},
  };

  // clang-format off
  for (PetscInt zi = 0; zi < shape.size[Z]; ++zi) {
  for (PetscInt yi = 0; yi < shape.size[Y]; ++yi) {
  for (PetscInt xi = 0; xi < shape.size[X]; ++xi) {
    Vector3R si = shape.electric(shape.s_p(zi, yi, xi));

    // Shifts of g'=g2 iteration
    for (PetscInt zj = 0; zj < shape.size[Z]; ++zj) {
    for (PetscInt yj = 0; yj < shape.size[Y]; ++yj) {
    for (PetscInt xj = 0; xj < shape.size[X]; ++xj) {
      Vector3R sj = shape.electric(shape.s_p(zj, yj, xj));

      if (si.abs_max() < shape_tolerance || sj.abs_max() < shape_tolerance)
        continue;

      PetscInt g = t_g(zi, yi, xi, zj, yj, xj);

      for (PetscInt c1 = 0; c1 < 3; ++c1) {
        coo_i[ind(g, X, c1)] = MatStencil{shape.start[Z] + zi, shape.start[Y] + yi, shape.start[X] + xi, X};
        coo_i[ind(g, Y, c1)] = MatStencil{shape.start[Z] + zi, shape.start[Y] + yi, shape.start[X] + xi, Y};
        coo_i[ind(g, Z, c1)] = MatStencil{shape.start[Z] + zi, shape.start[Y] + yi, shape.start[X] + xi, Z};

        coo_j[ind(g, c1, X)] = MatStencil{shape.start[Z] + zj, shape.start[Y] + yj, shape.start[X] + xj, X};
        coo_j[ind(g, c1, Y)] = MatStencil{shape.start[Z] + zj, shape.start[Y] + yj, shape.start[X] + xj, Y};
        coo_j[ind(g, c1, Z)] = MatStencil{shape.start[Z] + zj, shape.start[Y] + yj, shape.start[X] + xj, Z};

        for (PetscInt c2 = 0; c2 < 3; ++c2)
          coo_v[ind(g, c1, c2)] = si[c1] * sj[c2] * betaL * matB[c1][c2];
      }
    }}}
  }}}
  // clang-format on
  PetscFunctionReturnVoid();
#endif
}

// NOLINTEND(readability-function-cognitive-complexity)

}  // namespace ecsimcorr
