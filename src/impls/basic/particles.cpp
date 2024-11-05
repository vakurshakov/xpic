#include "particles.h"

#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/impls/basic/simulation.h"

namespace basic {

Particles::Particles(Simulation& simulation, const Sort_parameters& parameters)
  : interfaces::Particles(simulation.world_, parameters), simulation_(simulation)
{
  PetscFunctionBeginUser;
  /// @note This local current is local to each particle!
  /// It's can be useful for diagnosing it.
  DM da = simulation_.world_.da;
  PetscCallVoid(DMCreateLocalVector(da, &local_J));
  PetscFunctionReturnVoid();
}

Particles::~Particles()
{
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&local_J));
  PetscFunctionReturnVoid();
}


PetscErrorCode Particles::push()
{
  PetscFunctionBeginUser;
  DM da = simulation_.world_.da;
  PetscCall(DMGetLocalVector(da, &local_E));
  PetscCall(DMGetLocalVector(da, &local_B));

  PetscCall(DMGlobalToLocal(da, simulation_.E_, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, simulation_.B_, INSERT_VALUES, local_B));
  PetscCall(VecSet(local_J, 0.0));

  PetscCall(DMDAVecGetArrayRead(da, local_E, &E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayWrite(da, local_J, &J));

#pragma omp for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    Vector3R point_E;
    Vector3R point_B;

    const Node node(it->r);

    static Shape shape[2];
#pragma omp threadprivate(shape)

    shape[0].fill(node.g, node.r, false);
    shape[1].fill(node.g, node.r, true);
    interpolate(node.g, shape[0], shape[1], point_E, point_B);

    push(point_E, point_B, *it);

    const Node new_node(it->r);

    shape[0].fill(new_node.g, node.r, false);
    shape[1].fill(new_node.g, new_node.r, false);
    decompose(new_node.g, shape[0], shape[1], *it);
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_J, &J));

  PetscCall(DMLocalToGlobal(da, local_J, ADD_VALUES, simulation_.J_));

  PetscCall(DMRestoreLocalVector(da, &local_E));
  PetscCall(DMRestoreLocalVector(da, &local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}


void Particles::interpolate(const Vector3I& p_g, Shape& no, Shape& sh,
  Vector3R& point_E, Vector3R& point_B) const
{
  Simple_interpolation interpolation(shape_width, no, sh);
  interpolation.process(p_g, {{point_E, E}}, {{point_B, B}});
}


void Particles::push(
  const Vector3R& point_E, const Vector3R& point_B, Point& point) const
{
  PetscReal alpha = 0.5 * dt * charge(point);
  PetscReal m = mass(point);

  Vector3R& r = point.r;
  Vector3R& p = point.p;

  const Vector3R w = p + point_E * alpha;
  PetscReal energy = sqrt(m * m + w.dot(w));

  const Vector3R h = point_B * alpha / energy;
  const Vector3R s = h * 2.0 / (1.0 + h.dot(h));
  p = point_E * alpha + w * (1.0 - h.dot(s)) + w.cross(s) + h * (s.dot(w));

  energy = sqrt(m * m + p.dot(p));
  r += p * dt / energy;
}


void Particles::decompose(
  const Vector3I& p_g, Shape& old_shape, Shape& new_shape, const Point& point)
{
  const PetscReal alpha =
    charge(point) * density(point) / particles_number(point) / (6.0 * dt);

  Esirkepov_decomposition decomposition(
    shape_width, alpha, old_shape, new_shape);
  decomposition.process(p_g, J);
}

}  // namespace basic
