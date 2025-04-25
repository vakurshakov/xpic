#include "particles.h"

#include "src/algorithms/boris_push.h"
#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/diagnostics/particles_energy.h"
#include "src/impls/ecsim/simulation.h"

namespace ecsim {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world, parameters), simulation_(simulation)
{
  DM da = world.da;
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da, &local_currI));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &global_currI));
}

PetscErrorCode Particles::first_push()
{
  PetscFunctionBeginUser;
#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage) {
    for (auto& point : cell) {
      BorisPush::update_r(dt, point);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::fill_ecsim_current(PetscReal* coo_v)
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(world.da, local_currI, &currI));

  PetscInt prev_g = 0;
  PetscInt off = 0;

#pragma omp parallel for firstprivate(prev_g, off)
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    if (storage[g].empty())
      continue;

    simulation_.get_array_offset(prev_g, g, off);
    prev_g = g;

    PetscReal* coo_cv = coo_v + off;
    for (const auto& point : storage[g]) {
      Shape shape;
      shape.setup(point.r, shape_radius1, shape_func1);

      Vector3R B_p;
      SimpleInterpolation interpolation(shape);
      interpolation.process({}, {{B_p, B}});

      decompose_ecsim_current(shape, point, B_p, coo_cv);
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(world.da, local_currI, &currI));
  PetscCall(DMLocalToGlobal(world.da, local_currI, ADD_VALUES, global_currI));
  PetscCall(VecAXPY(simulation_.currI, 1.0, global_currI));
  PetscFunctionReturn(PETSC_SUCCESS);
}

constexpr PetscInt ind(PetscInt g, PetscInt c1, PetscInt c2)
{
  return g * POW2(3) + (c1 * 3 + c2);
}

/// @note Also decomposes `Simulation::matL`
void Particles::decompose_ecsim_current(
  const Shape& shape, const Point& point, const Vector3R& B_p, PetscReal* coo_v)
{
  PetscFunctionBeginUser;
  const auto& [r, v] = point;

  Vector3R b = 0.5 * dt * q_m(point) * B_p;

  PetscReal betaI = qn_Np(point) / (1.0 + b.squared());
  PetscReal betaL = q_m(point) * betaI;

  Vector3R I_p = betaI * (v + v.cross(b) + b * v.dot(b));

  SimpleDecomposition decomposition(shape, I_p);
  PetscCallVoid(decomposition.process(currI));

  constexpr PetscInt shr = 1;
  constexpr PetscInt shw = 2 * shr + 1;
  constexpr PetscReal shape_tolerance = 1e-10;

  /// @note It is an offset from particle `shape` indexing into `coo_v` one.
  const Vector3I off{
    shape.start[X] - (FLOOR_STEP(r.x(), dx) - shr),
    shape.start[Y] - (FLOOR_STEP(r.y(), dy) - shr),
    shape.start[Z] - (FLOOR_STEP(r.z(), dz) - shr),
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

    return  //
      ((vg1[Z] * shw + vg1[Y]) * shw + vg1[X]) * POW3(shw) +
      ((vg2[Z] * shw + vg2[Y]) * shw + vg2[X]);
  };

  const PetscReal matB[Vector3R::dim][Vector3R::dim]{
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

      for (PetscInt c1 = 0; c1 < Vector3R::dim; ++c1)
        for (PetscInt c2 = 0; c2 < Vector3R::dim; ++c2)
          coo_v[ind(gg, c1, c2)] += s1[c1] * s2[c2] * betaL * matB[c1][c2];
    }
  }
  PetscFunctionReturnVoid();
}

PetscErrorCode Particles::second_push()
{
  PetscFunctionBeginUser;
#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage) {
    for (auto& point : cell) {
      Shape shape;
      shape.setup(point.r, shape_radius1, shape_func1);

      Vector3R E_p;
      Vector3R B_p;
      SimpleInterpolation interpolation(shape);
      interpolation.process({{E_p, E}}, {{B_p, B}});

      BorisPush push(q_m(point), E_p, B_p);
      push.update_vEB(dt, point);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(local_currI, 0.0));
  PetscCall(VecSet(global_currI, 0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&local_currI));
  PetscCall(VecDestroy(&global_currI));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace ecsim
