#include "particles.h"

#include "src/algorithms/boris_push.h"
#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/diagnostics/particles_energy.h"
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
  PetscLogEventRegister("second_push", classid, &events[1]);
  PetscLogEventRegister("final_update", classid, &events[2]);
  PetscFunctionReturnVoid();
}


PetscErrorCode Particles::first_push()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(world.da, local_currJe, &currJe));

  PetscLogEventBegin(events[0], local_currJe, 0, 0, 0);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage) {
    for (auto& point : cell) {
      const Vector3R old_r = point.r;

      BorisPush push;
      push.update_r((0.5 * dt), point, *this);

      Shape shape;
      shape.setup(old_r, point.r, shape_radius2, shape_func2);
      decompose_esirkepov_current(shape, point);
    }
  }

  PetscLogEventEnd(events[0], local_currJe, 0, 0, 0);

  PetscCall(DMDAVecRestoreArrayWrite(world.da, local_currJe, &currJe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

void Particles::fill_ecsim_current(PetscInt g, PetscReal* coo_v)
{
  for (const auto& point : storage[g]) {
    Shape shape;
    shape.setup(point.r, shape_radius1, shape_func1);

    Vector3R B_p;
    SimpleInterpolation interpolation(shape);
    interpolation.process({}, {{B_p, B}});

    decompose_ecsim_current(shape, point, B_p, coo_v);
  }
}

void Particles::fill_matrix_indices(
  PetscInt g, MatStencil* coo_i, MatStencil* coo_j)
{
  PetscFunctionBeginUser;
  constexpr PetscInt shw = 3;

  const Vector3I vg{
    world.start[X] + g % world.size[X],
    world.start[Y] + (g / world.size[X]) % world.size[Y],
    world.start[Z] + (g / world.size[X]) / world.size[Y],
  };

  for (PetscInt g1 = 0; g1 < POW3(shw); ++g1) {
    for (PetscInt g2 = 0; g2 < POW3(shw); ++g2) {
      PetscInt gg = g1 * POW3(shw) + g2;

      Vector3I vg1{
        vg[X] + g1 % shw - 1,
        vg[Y] + (g1 / shw) % shw - 1,
        vg[Z] + (g1 / shw) / shw - 1,
      };

      Vector3I vg2{
        vg[X] + g2 % shw - 1,
        vg[Y] + (g2 / shw) % shw - 1,
        vg[Z] + (g2 / shw) / shw - 1,
      };

      for (PetscInt c = 0; c < 3; ++c) {
        coo_i[ind(gg, X, c)] = MatStencil{vg1[Z], vg1[Y], vg1[X], X};
        coo_i[ind(gg, Y, c)] = MatStencil{vg1[Z], vg1[Y], vg1[X], Y};
        coo_i[ind(gg, Z, c)] = MatStencil{vg1[Z], vg1[Y], vg1[X], Z};

        coo_j[ind(gg, c, X)] = MatStencil{vg2[Z], vg2[Y], vg2[X], X};
        coo_j[ind(gg, c, Y)] = MatStencil{vg2[Z], vg2[Y], vg2[X], Y};
        coo_j[ind(gg, c, Z)] = MatStencil{vg2[Z], vg2[Y], vg2[X], Z};
      }
    }
  }
  PetscFunctionReturnVoid();
}

/// @note Also decomposes `Simulation::matL`
void Particles::decompose_ecsim_current(
  const Shape& shape, const Point& point, const Vector3R& B_p, PetscReal* coo_v)
{
  PetscFunctionBeginUser;
  const Vector3R& v = point.p;

  Vector3R b = 0.5 * dt * charge(point) / mass(point) * B_p;

  PetscReal betaI = density(point) * charge(point) /
    (particles_number(point) * (1.0 + b.squared()));

  PetscReal betaL = charge(point) / mass(point) * betaI;

  Vector3R I_p = betaI * (v + v.cross(b) + b * v.dot(b));

  SimpleDecomposition decomposition(shape, I_p);
  PetscCallVoid(decomposition.process(currI));

  constexpr PetscReal shape_tolerance = 1e-10;
  constexpr PetscInt shw = 3;

  /// @note It is an offset from particle `shape` indexing into `coo_v` one.
  const Vector3I off{
    shape.start[X] - ((PetscInt)(point.x() / dx) - 1),
    shape.start[Y] - ((PetscInt)(point.y() / dy) - 1),
    shape.start[Z] - ((PetscInt)(point.z() / dz) - 1),
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

    return //
      ((vg1[Z] * shw + vg1[Y]) * shw + vg1[X]) * POW3(shw) +
      ((vg2[Z] * shw + vg2[Y]) * shw + vg2[X]);
  };

  const PetscReal matB[3][3]{
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

      for (PetscInt c1 = 0; c1 < 3; ++c1)
        for (PetscInt c2 = 0; c2 < 3; ++c2)
          coo_v[ind(gg, c1, c2)] += s1[c1] * s2[c2] * betaL * matB[c1][c2];
    }
  }
  PetscFunctionReturnVoid();
}


PetscErrorCode Particles::second_push()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(world.da, local_currJe, &currJe));

  PetscLogEventBegin(events[1], 0, 0, 0, 0);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage) {
    for (auto& point : cell) {
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
  }

  PetscLogEventEnd(events[1], 0, 0, 0, 0);

  PetscCall(DMLocalToGlobal(world.da, local_currJe, ADD_VALUES, global_currJe));
  PetscCall(VecAXPY(simulation_.currJe, 1.0, global_currJe));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::final_update()
{
  PetscFunctionBeginUser;
  PetscCall(MatMultAdd(simulation_.matL, simulation_.Ep, global_currI, global_currI));
  PetscCall(VecDot(global_currI, simulation_.Ep, &pred_w));
  PetscCall(VecDot(global_currJe, simulation_.Ec, &corr_w));

  PetscReal K0 = energy;
  PetscLogEventBegin(events[2], 0, 0, 0, 0);

  calculate_energy();
  PetscReal K = energy;

  PetscReal lambda2 = 1.0 + dt * (corr_w - pred_w) / K;
  PetscReal lambda = std::sqrt(lambda2);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage)
    for (auto& point : cell)
      point.p *= lambda;

  PetscLogEventEnd(events[2], 0, 0, 0, 0);

  lambda_dK = (lambda2 - 1.0) * K;
  pred_dK = K - K0;
  corr_dK = lambda2 * K - K0;
  energy = lambda2 * K;

  LOG("  Velocity renormalization for \"{}\"", parameters.sort_name);
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

Particles::~Particles()
{
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&local_currI));
  PetscCallVoid(VecDestroy(&local_currJe));
  PetscCallVoid(VecDestroy(&global_currI));
  PetscCallVoid(VecDestroy(&global_currJe));
  PetscFunctionReturnVoid();
}

PetscErrorCode Particles::calculate_energy()
{
  PetscFunctionBeginUser;
  energy = 0.0;

  const PetscReal m = parameters.m;
  const PetscInt Np = parameters.Np;

#pragma omp parallel for reduction(+ : energy), \
  schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage)
    for (auto& point : cell)
      energy += ParticlesEnergy::get(point.p, m, Np);

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
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


}  // namespace ecsimcorr
