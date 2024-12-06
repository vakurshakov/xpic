#include "particles.h"

#include "src/algorithms/boris_push.h"
#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/impls/basic/simulation.h"

namespace basic {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world_, parameters), simulation_(simulation)
{
  PetscFunctionBeginUser;
  /// @note This local current is local to each particle!
  /// It's can be useful for diagnosing it.
  DM da = world_.da;
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
  DM da = world_.da;
  PetscCall(DMGetLocalVector(da, &local_E));
  PetscCall(DMGetLocalVector(da, &local_B));

  PetscCall(DMGlobalToLocal(da, simulation_.E_, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, simulation_.B_, INSERT_VALUES, local_B));
  PetscCall(VecSet(local_J, 0.0));

  PetscCall(DMDAVecGetArrayRead(da, local_E, reinterpret_cast<void*>(&E)));
  PetscCall(DMDAVecGetArrayRead(da, local_B, reinterpret_cast<void*>(&B)));
  PetscCall(DMDAVecGetArrayWrite(da, local_J, reinterpret_cast<void*>(&J)));

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_) {
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

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, reinterpret_cast<void*>(&E)));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, reinterpret_cast<void*>(&B)));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_J, reinterpret_cast<void*>(&J)));

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
