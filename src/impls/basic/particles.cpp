#include "particles.h"

#include "src/algorithms/boris_push.h"
#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/impls/basic/simulation.h"

namespace basic {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world, parameters), simulation_(simulation)
{
  PetscFunctionBeginUser;
  PetscCallVoid(DMCreateLocalVector(world.da, &local_J));
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
  DM da = world.da;
  PetscCall(DMGetLocalVector(da, &local_E));
  PetscCall(DMGetLocalVector(da, &local_B));

  PetscCall(DMGlobalToLocal(da, simulation_.E_, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, simulation_.B_, INSERT_VALUES, local_B));
  PetscCall(VecSet(local_J, 0.0));

  PetscCall(DMDAVecGetArrayRead(da, local_E, &E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayWrite(da, local_J, &J));

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage) {
    for (auto& point : cell) {
      const Vector3R old_r = point.r;

      Shape shape;
      shape.setup(point.r);

      Vector3R E_p;
      Vector3R B_p;
      interpolate(shape, E_p, B_p);

      push(E_p, B_p, point);

      shape.setup(old_r, point.r);
      decompose(shape, point);
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_J, &J));

  PetscCall(DMLocalToGlobal(da, local_J, ADD_VALUES, simulation_.J_));

  PetscCall(DMRestoreLocalVector(da, &local_E));
  PetscCall(DMRestoreLocalVector(da, &local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}


void Particles::interpolate(const Shape& shape, Vector3R& E_p, Vector3R& B_p) const
{
  SimpleInterpolation interpolation(shape);
  interpolation.process({{E_p, E}}, {{B_p, B}});
}


void Particles::push(const Vector3R& E_p, const Vector3R& B_p, Point& point) const
{
  BorisPush push(dt, E_p, B_p);
  push.process_rel(point, *this);
}


void Particles::decompose(const Shape& shape, const Point& point)
{
  const PetscReal alpha =
    charge(point) * density(point) / (particles_number(point) * (6.0 * dt));

  EsirkepovDecomposition decomposition(shape, alpha);
  decomposition.process(J);
}

}  // namespace basic
