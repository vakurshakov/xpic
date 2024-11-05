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

PetscErrorCode Particles::reset()
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
    decompose_identity_current(node.r, shape[0], shape[1], point, point_B);
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_currI, &currI));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_currJe, &currJe));

  PetscCall(DMLocalToGlobal(da, local_currI, ADD_VALUES, simulation_.currI));
  PetscCall(DMLocalToGlobal(da, local_currJe, ADD_VALUES, simulation_.currJe));

  PetscCall(DMRestoreLocalVector(da, &local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}


void Particles::decompose_esirkepov_current(const Vector3I& p_g,
  const Shape& old_shape, const Shape& new_shape, const Point& point)
{
  const PetscReal alpha =
    charge(point) * density(point) / particles_number(point) / (6.0 * dt);

  Esirkepov_decomposition decomposition(shape_width2, alpha, old_shape, new_shape);
  decomposition.process(p_g, currJe);
}

void Particles::first_interpolate(const Vector3I& p_g, const Shape& no,
  const Shape& sh, Vector3R& point_B) const
{
  Simple_interpolation interpolation(shape_width1, no, sh);
  interpolation.process(p_g, {}, {{point_B, B}});
}

void Particles::decompose_identity_current(const Vector3R& p_r, const Shape& no,
  const Shape& sh, const Point& point, const Vector3R& point_B)
{
  const Vector3R& v = point.p;

  PetscReal alpha1 = 0.5 * dt * charge(point) / mass(point) * point_B.length();
  PetscReal alpha2 = POW2(alpha1);

  PetscReal beta = charge(point) / (particles_number(point) * (1.0 + alpha1));

  Vector3R h = point_B.normalized();
  Vector3R J_p = beta * (v + alpha1 * v.cross(h) + alpha2 * v.dot(h) * h);

  Simple_decomposition decomposition(shape_width1, J_p, no, sh);
  decomposition.process(p_r, currI);
}

}  // namespace ecsimcorr
