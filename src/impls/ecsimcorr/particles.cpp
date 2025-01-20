#include "particles.h"

#include "src/algorithms/boris_push.h"
#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/impls/ecsimcorr/simulation.h"

namespace ecsimcorr {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world_, parameters), simulation_(simulation)
{
  PetscFunctionBeginUser;
  DM da = world_.da;
  PetscCallVoid(DMCreateLocalVector(da, &local_currI));
  PetscCallVoid(DMCreateLocalVector(da, &local_currJe));
  PetscCallVoid(DMCreateGlobalVector(da, &global_currI));
  PetscCallVoid(DMCreateGlobalVector(da, &global_currJe));

  PetscClassIdRegister("ecsimcorr::Particles", &classid);
  PetscLogEventRegister("first_push", classid, &events[0]);
  PetscLogEventRegister("second_push", classid, &events[1]);
  PetscLogEventRegister("final_update", classid, &events[2]);
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
  Vec local_B;
  Vector3R*** B;

  DM da = world_.da;
  PetscCall(DMGetLocalVector(da, &local_B));
  PetscCall(DMGlobalToLocal(da, simulation_.B, INSERT_VALUES, local_B));

  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayWrite(da, local_currI, &currI));
  PetscCall(DMDAVecGetArrayWrite(da, local_currJe, &currJe));

  PetscLogEventBegin(events[0], 0, 0, 0, 0);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_) {
    const Vector3R old_r = point.r;

    BorisPush push;
    push.update_r((0.5 * dt), point, *this);

    Shape shape;
    shape.setup(point.r, shape_radius1, shape_func1);

    Vector3R B_p;
    SimpleInterpolation interpolation(shape);
    interpolation.process({}, {{B_p, B}});

    decompose_identity_current(shape, point, B_p);

    shape.setup(old_r, point.r, shape_radius2, shape_func2);
    decompose_esirkepov_current(shape, point);
  }

  PetscLogEventEnd(events[0], 0, 0, 0, 0);

  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_currI, &currI));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_currJe, &currJe));
  PetscCall(DMRestoreLocalVector(da, &local_B));

  PetscCall(DMLocalToGlobal(da, local_currI, ADD_VALUES, global_currI));
  PetscCall(VecAXPY(simulation_.currI, 1.0, global_currI));

  PetscCall(DMLocalToGlobal(da, local_currJe, ADD_VALUES, global_currJe));
  PetscCall(VecAXPY(simulation_.currJe, 1.0, global_currJe));

  PetscCall(MatAssemblyBegin(simulation_.matL, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(simulation_.matL, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::second_push()
{
  PetscFunctionBeginUser;
  Vec local_E;
  Vec local_B;
  Vector3R*** E;
  Vector3R*** B;

  DM da = world_.da;
  PetscCall(DMGetLocalVector(da, &local_E));
  PetscCall(DMGetLocalVector(da, &local_B));
  PetscCall(DMGlobalToLocal(da, simulation_.Ep, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, simulation_.B, INSERT_VALUES, local_B));

  PetscCall(DMDAVecGetArrayRead(da, local_E, &E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayWrite(da, local_currJe, &currJe));

  PetscLogEventBegin(events[1], 0, 0, 0, 0);

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
  }

  PetscLogEventEnd(events[1], 0, 0, 0, 0);

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
  // PetscCall(MatMultAdd(simulation_.matL, simulation_.Ep, global_currI, global_currI));
  PetscCall(VecDot(global_currI, simulation_.Ep, &pred_w));
  PetscCall(VecDot(global_currJe, simulation_.Ec, &corr_w));

  PetscReal K0 = energy;
  PetscReal K = 0.0;
  PetscLogEventBegin(events[2], 0, 0, 0, 0);

#pragma omp parallel for reduction(+ : K), \
  schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_)
    K += 0.5 * (mass(point) / particles_number(point)) * point.p.squared();

  PetscReal lambda2 = 1.0 + dt * (corr_w - pred_w) / K;
  PetscReal lambda = std::sqrt(lambda2);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_)
    point.p *= lambda;

  PetscLogEventEnd(events[2], 0, 0, 0, 0);

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
  const PetscReal alpha =
    charge(point) * density(point) / (particles_number(point) * (6.0 * dt));

  EsirkepovDecomposition decomposition(shape, alpha);
  PetscCallVoid(decomposition.process(currJe));
}


// NOLINTBEGIN(readability-function-cognitive-complexity)

/// @note Also decomposes `Simulation::matL`
void Particles::decompose_identity_current(
  const Shape& shape, const Point& point, const Vector3R& B_p)
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


  /// @todo Combine it with `Simple_decomposition::process()`?
  Mat matL = simulation_.matL;

  const PetscInt m = shape.size.elements_product();
  const PetscInt n = m;

  std::vector<MatStencil> idxm(m);
  std::vector<MatStencil> idxn(n);
  std::vector<PetscReal> values(static_cast<std::size_t>(m * n * POW2(3)), 0);

  constexpr PetscReal shape_tolerance = 1e-10;

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
  for (PetscInt z1 = 0; z1 < shape.size[Z]; ++z1) {
  for (PetscInt y1 = 0; y1 < shape.size[Y]; ++y1) {
  for (PetscInt x1 = 0; x1 < shape.size[X]; ++x1) {
    PetscInt i = shape.s_p(z1, y1, x1);
    Vector3R si = shape.electric(i);

    idxm[i] = MatStencil{
      shape.start[Z] + z1,
      shape.start[Y] + y1,
      shape.start[X] + x1,
    };

    // Shifts of g'=g2 iteration
    for (PetscInt z2 = 0; z2 < shape.size[Z]; ++z2) {
    for (PetscInt y2 = 0; y2 < shape.size[Y]; ++y2) {
    for (PetscInt x2 = 0; x2 < shape.size[X]; ++x2) {
      PetscInt j = shape.s_p(z2, y2, x2);
      Vector3R sj = shape.electric(j);

      idxn[j] = MatStencil{
        shape.start[Z] + z2,
        shape.start[Y] + y2,
        shape.start[X] + x2,
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
}

// NOLINTEND(readability-function-cognitive-complexity)

}  // namespace ecsimcorr
