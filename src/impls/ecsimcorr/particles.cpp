#include "particles.h"

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
  PetscCall(DMGlobalToLocalBegin(da, simulation_.B, INSERT_VALUES, local_B));

  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayWrite(da, local_currI, &currI));
  PetscCall(DMDAVecGetArrayWrite(da, local_currJe, &currJe));

#pragma omp for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_) {
    /// @todo We _must_ save particle initial point.p!
    Vector3R old_nr = Node::make_r(point.r);
    point.r += point.p * dt;

    Shape shape[2];
    const Node node(point.r);

    shape[0].fill(node.g, old_nr, false, shape_func2, shape_width2);
    shape[1].fill(node.g, node.r, false, shape_func2, shape_width2);
    decompose_esirkepov_current(node.g, shape[0], shape[1], point);

    Vector3R point_B;
    shape[0].fill(node.g, node.r, false, shape_func1, shape_width1);
    shape[1].fill(node.g, node.r, true, shape_func1, shape_width1);
    first_interpolate(node.g, shape[0], shape[1], point_B);
    decompose_identity_current(node.g, shape[0], shape[1], point, point_B);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::final_update()
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}


void Particles::decompose_esirkepov_current(const Vector3I& p_g,
  const Shape& old_shape, const Shape& new_shape, const Point& point)
{
  const PetscReal alpha =
    charge(point) * density(point) / particles_number(point) / (6.0 * dt);

  Esirkepov_decomposition decomposition(
    shape_width2, alpha, old_shape, new_shape);
  PetscCallVoid(decomposition.process(p_g, currJe));
}

void Particles::first_interpolate(const Vector3I& p_g, const Shape& no,
  const Shape& sh, Vector3R& point_B) const
{
  Simple_interpolation interpolation(shape_width1, no, sh);
  PetscCallVoid(interpolation.process(p_g, {}, {{point_B, B}}));
}

/// @note Also decomposes `Simulation::matL`
void Particles::decompose_identity_current(const Vector3I& p_g, const Shape& no,
  const Shape& sh, const Point& point, const Vector3R& point_B)
{
  PetscFunctionBeginHot;
  const Vector3R& v = point.p;

  PetscReal alpha = 0.5 * dt * charge(point) / mass(point) * point_B.length();
  PetscReal alpha2 = POW2(alpha);

  PetscReal betaI = charge(point) / (particles_number(point) * (1.0 + alpha));
  PetscReal betaL = charge(point) / mass(point) * betaI;

  Vector3R h = point_B.normalized();
  Vector3R J_p = betaI * (v + alpha * v.cross(h) + alpha2 * v.dot(h) * h);

  Simple_decomposition decomposition(shape_width1, J_p, no, sh);
  PetscCallVoid(decomposition.process(p_g, currI));


  /// @todo Combine it with `Simple_decomposition::process()`?
  Mat matL = simulation_.matL;

  constexpr PetscInt m = POW3(shape_width1);
  constexpr PetscInt n = POW3(shape_width1);
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
  for (PetscInt z1 = 0; z1 < shape_width1; ++z1) {
  for (PetscInt y1 = 0; y1 < shape_width1; ++y1) {
  for (PetscInt x1 = 0; x1 < shape_width1; ++x1) {
    i = Shape::index(z1, y1, x1);
    idxm[i] = MatStencil{ p_g[Z] + z1, p_g[Y] + y1, p_g[X] + x1 };

    Vector3R s1{
      no(i, Z) * no(i, Y) * sh(i, X),
      no(i, Z) * sh(i, Y) * no(i, X),
      sh(i, Z) * no(i, Y) * no(i, X),
    };

    // Shifts of g'=g2 iteration
    for (PetscInt z2 = 0; z2 < shape_width1; ++z2) {
    for (PetscInt y2 = 0; y2 < shape_width1; ++y2) {
    for (PetscInt x2 = 0; x2 < shape_width1; ++x2) {
      j = Shape::index(z2, y2, x2);
      idxn[j] = MatStencil{ p_g[Z] + z2, p_g[Y] + y2, p_g[X] + x2 };

      Vector3R s2{
        no(j, Z) * no(j, Y) * sh(j, X),
        no(j, Z) * sh(j, Y) * no(j, X),
        sh(j, Z) * no(j, Y) * no(j, X),
      };

      values[ind(i, j, X, X)] = s1[X] * s2[X] * betaL * (1.0 + alpha2 * h[X] * h[X]);
      values[ind(i, j, X, Y)] = s1[X] * s2[Y] * betaL * alpha * (+h[Z] + alpha * h[X] * h[Y]);
      values[ind(i, j, X, Z)] = s1[X] * s2[Z] * betaL * alpha * (-h[Y] + alpha * h[X] * h[Z]);

      values[ind(i, j, Y, Y)] = s1[Y] * s2[Y] * betaL * (1.0 + alpha2 * h[Y] * h[Y]);
      values[ind(i, j, Y, X)] = s1[Y] * s2[X] * betaL * alpha * (-h[Z] + alpha * h[Y] * h[X]);
      values[ind(i, j, Y, Z)] = s1[Y] * s2[Z] * betaL * alpha * (+h[X] + alpha * h[Y] * h[Z]);

      values[ind(i, j, Z, Z)] = s1[Z] * s2[Z] * betaL * (1.0 + alpha2 * h[Z] * h[Z]);
      values[ind(i, j, Z, X)] = s1[Z] * s2[X] * betaL * alpha * (+h[Y] + alpha * h[X] * h[Z]);
      values[ind(i, j, Z, Y)] = s1[Z] * s2[Y] * betaL * alpha * (-h[X] + alpha * h[Y] * h[Z]);
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
