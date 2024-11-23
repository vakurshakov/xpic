#include "particles.h"

#include "src/algorithms/boris_push.h"
#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/impls/ecsimcorr/simulation.h"

namespace ecsimcorr {

Particles::Particles(Simulation& simulation, const Sort_parameters& parameters)
  : interfaces::Particles(simulation.world_, parameters), simulation_(simulation)
{
  PetscFunctionBeginUser;
  DM da = simulation_.world_.da;
  PetscCallVoid(DMCreateLocalVector(da, &local_currI));
  PetscCallVoid(DMCreateLocalVector(da, &local_currJe));
  PetscFunctionReturnVoid();
}

Particles::~Particles()
{
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&local_currI));
  PetscCallVoid(VecDestroy(&local_currJe));
  PetscFunctionReturnVoid();
}

PetscErrorCode Particles::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(local_currI, 0.0));
  PetscCall(VecSet(local_currJe, 0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::first_push()
{
  PetscFunctionBeginUser;
  DM da = simulation_.world_.da;
  PetscCall(DMGetLocalVector(da, &local_B));
  PetscCall(DMGlobalToLocal(da, simulation_.B, INSERT_VALUES, local_B));

  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayWrite(da, local_currI, &currI));
  PetscCall(DMDAVecGetArrayWrite(da, local_currJe, &currJe));

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_) {
    const Vector3R old_r = point.r;
    point.r += point.p * (0.5 * dt);

    Shape shape;
    shape.setup(point.r, shape_radius1, shape_func1);

    Vector3R B_p;
    Simple_interpolation interpolation(shape);
    interpolation.process({}, {{B_p, B}});

    decompose_identity_current(shape, point, B_p);

    shape.setup(old_r, point.r, shape_radius2, shape_func2);
    decompose_esirkepov_current(shape, point);
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_currI, &currI));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_currJe, &currJe));

  PetscCall(DMLocalToGlobal(da, local_currI, ADD_VALUES, simulation_.currI));
  PetscCall(DMLocalToGlobal(da, local_currJe, ADD_VALUES, simulation_.currJe));

  PetscCall(DMRestoreLocalVector(da, &local_B));

  PetscCall(MatAssemblyBegin(simulation_.matL, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(simulation_.matL, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::second_push()
{
  PetscFunctionBeginUser;
  DM da = simulation_.world_.da;
  PetscCall(DMGetLocalVector(da, &local_E));
  PetscCall(DMGetLocalVector(da, &local_B));
  PetscCall(DMGlobalToLocal(da, simulation_.En, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, simulation_.B, INSERT_VALUES, local_B));

  PetscCall(DMDAVecGetArrayRead(da, local_E, &E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayWrite(da, local_currJe, &currJe));

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_) {
    const Vector3R old_r = point.r;

    Shape shape;
    shape.setup(point.r, shape_radius1, shape_func1);

    Vector3R E_p, B_p;
    Simple_interpolation interpolation(shape);
    interpolation.process({{E_p, E}}, {{B_p, B}});

    Boris_push push((0.5 * dt), E_p, B_p);
    push.process(point, *this);

    shape.setup(old_r, point.r, shape_radius2, shape_func2);
    decompose_esirkepov_current(shape, point);
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_currJe, &currJe));

  PetscCall(DMLocalToGlobal(da, local_currJe, ADD_VALUES, simulation_.currJe));

  PetscCall(DMRestoreLocalVector(da, &local_E));
  PetscCall(DMRestoreLocalVector(da, &local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::final_update()
{
  PetscFunctionBeginUser;
  PetscReal w1 = simulation_.w1;
  PetscReal w2 = simulation_.w2;
  PetscReal wp = 0.0;

#pragma omp parallel for reduction(+ : wp), \
  schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_)
    wp += 0.5 * (mass(point) / particles_number(point)) * point.p.squared();

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_)
    point.p *= 1.0 + dt * (w2 - w1) / wp;

  PetscFunctionReturn(PETSC_SUCCESS);
}


void Particles::decompose_esirkepov_current(const Shape& shape, const Point& point)
{
  const PetscReal alpha =
    charge(point) * density(point) / (particles_number(point) * (3.0 * dt));

  Esirkepov_decomposition decomposition(shape, alpha);
  PetscCallVoid(decomposition.process(currJe));
}


/// @note Also decomposes `Simulation::matL`
void Particles::decompose_identity_current(
  const Shape& shape, const Point& point, const Vector3R& B_p)
{
  PetscFunctionBeginHot;
  const Vector3R& v = point.p;

  Vector3R b = 0.5 * dt * charge(point) / mass(point) * B_p;
  PetscReal alpha2 = b.squared();

  PetscReal betaI = charge(point) / (particles_number(point) * (1.0 + alpha2));
  PetscReal betaL = charge(point) / mass(point) * betaI;

  Vector3R I_p = betaI * (v + v.cross(b) + b * v.dot(b));

  Simple_decomposition decomposition(shape, I_p);
  PetscCallVoid(decomposition.process(currI));


  /// @todo Combine it with `Simple_decomposition::process()`?
  Mat matL = simulation_.matL;

  constexpr PetscInt m = POW3(static_cast<PetscInt>(2.0 * shape_radius1) + 1);
  constexpr PetscInt n = POW3(static_cast<PetscInt>(2.0 * shape_radius1) + 1);
  MatStencil idxm[m], idxn[n];
  PetscInt i, j;

  PetscReal values[m * n * POW2(3)];

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
    i = shape.s_p(z1, y1, x1);
    Vector3R s1 = shape.electric(i);

    idxm[i] = MatStencil{
      shape.start[Z] + z1,
      shape.start[Y] + y1,
      shape.start[X] + x1,
    };

    // Shifts of g'=g2 iteration
    for (PetscInt z2 = 0; z2 < shape.size[Z]; ++z2) {
    for (PetscInt y2 = 0; y2 < shape.size[Y]; ++y2) {
    for (PetscInt x2 = 0; x2 < shape.size[X]; ++x2) {
      j = shape.s_p(z2, y2, x2);
      Vector3R s2 = shape.electric(j);

      idxn[j] = MatStencil{
        shape.start[Z] + z2,
        shape.start[Y] + y2,
        shape.start[X] + x2,
      };

      values[ind(i, j, X, X)] = s1[X] * s2[X] * betaL * (1.0   + b[X] * b[X]);
      values[ind(i, j, X, Y)] = s1[X] * s2[Y] * betaL * (+b[Z] + b[X] * b[Y]);
      values[ind(i, j, X, Z)] = s1[X] * s2[Z] * betaL * (-b[Y] + b[X] * b[Z]);

      values[ind(i, j, Y, Y)] = s1[Y] * s2[Y] * betaL * (1.0   + b[Y] * b[Y]);
      values[ind(i, j, Y, X)] = s1[Y] * s2[X] * betaL * (-b[Z] + b[Y] * b[X]);
      values[ind(i, j, Y, Z)] = s1[Y] * s2[Z] * betaL * (+b[X] + b[Y] * b[Z]);

      values[ind(i, j, Z, Z)] = s1[Z] * s2[Z] * betaL * (1.0   + b[Z] * b[Z]);
      values[ind(i, j, Z, X)] = s1[Z] * s2[X] * betaL * (+b[Y] + b[X] * b[Z]);
      values[ind(i, j, Z, Y)] = s1[Z] * s2[Y] * betaL * (-b[X] + b[Y] * b[Z]);
    }}}  // g'=g2
  }}}  // g=g1
  // clang-format on

#pragma omp critical
  {
    // cannot use `PetscCall()`, omp section cannot be broken by return statement
    MatSetValuesBlockedStencil(matL, m, idxm, n, idxn, values, ADD_VALUES);
  }
  PetscFunctionReturnVoid();
}

}  // namespace ecsimcorr
