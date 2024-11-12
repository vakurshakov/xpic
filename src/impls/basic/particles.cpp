#include "particles.h"

#include "src/algorithms/boris_push.h"
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

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_) {
    Vector3R E_p;
    Vector3R B_p;

    Vector3R old_nr = Node::make_r(point.r);
    Node node(point.r);

    static Shape shape[2];
#pragma omp threadprivate(shape)

    shape[0].fill(node.g, node.r, false);
    shape[1].fill(node.g, node.r, true);
    interpolate(node.g, shape[0], shape[1], E_p, B_p);

    push(E_p, B_p, point);
    node.update(point.r);

    shape[0].fill(node.g, old_nr, false);
    shape[1].fill(node.g, node.r, false);
    decompose(node.g, shape[0], shape[1], point);
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_J, &J));

  PetscCall(DMLocalToGlobal(da, local_J, ADD_VALUES, simulation_.J_));

  PetscCall(DMRestoreLocalVector(da, &local_E));
  PetscCall(DMRestoreLocalVector(da, &local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}


void Particles::interpolate(
  const Vector3I& p_g, Shape& no, Shape& sh, Vector3R& E_p, Vector3R& B_p) const
{
  Simple_interpolation interpolation(shape_width, no, sh);
  interpolation.process(p_g, {{E_p, E}}, {{B_p, B}});
}


void Particles::push(const Vector3R& E_p, const Vector3R& B_p, Point& point) const
{
  Boris_push push(dt, E_p, B_p);
  push.process_rel(point, *this);
}


void Particles::decompose(
  const Vector3I& p_g, Shape& old_shape, Shape& new_shape, const Point& point)
{
  const PetscReal alpha =
    charge(point) * density(point) / (particles_number(point) * (6.0 * dt));

  Esirkepov_decomposition decomposition(shape_width, alpha, old_shape, new_shape);
  decomposition.process(p_g, J);
}

}  // namespace basic
