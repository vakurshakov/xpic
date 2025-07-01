#include "charge_conservation.h"

#include "src/utils/configuration.h"
#include "src/utils/operators.h"
#include "src/utils/shape.h"

ParticlesChargeDensity::ParticlesChargeDensity(
  const interfaces::Particles& particles)
  : DistributionMoment(particles)
{
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Comm_dup(PETSC_COMM_WORLD, &comm_));

  FieldView::Region region{
    .dim = 3,
    .dof = 1,
    .start = Vector4I(0, 0, 0, 0),
    .size = Vector4I(geom_nx, geom_ny, geom_nz, 1),
  };

  PetscCallAbort(PETSC_COMM_WORLD, set_data_views(region));
}

PetscErrorCode ParticlesChargeDensity::set_data_views(
  const FieldView::Region& region)
{
  PetscFunctionBeginUser;
  PetscCall(set_local_da(region));
  PetscCall(DMCreateLocalVector(da_, &local_));
  PetscCall(DMCreateGlobalVector(da_, &field_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct ParticlesChargeDensity::Shape {
  Shape() = default;

  static constexpr PetscReal shr = shape_radius;
  static constexpr PetscInt shw = (PetscInt)(2.0 * shr);
  static constexpr PetscInt shm = POW3(shw);
  static constexpr PetscReal (&sfunc)(PetscReal) = shape_function;

  Vector3I start;
  PetscReal cache[shm];

  void setup(const Vector3R& r)
  {
    Vector3R p_r = ::Shape::make_r(r);

    start = Vector3I{
      (PetscInt)(std::ceil(p_r[X] - shr)),
      (PetscInt)(std::ceil(p_r[Y] - shr)),
      (PetscInt)(std::ceil(p_r[Z] - shr)),
    };

#pragma omp simd
    for (PetscInt i = 0; i < shm; ++i) {
      auto g_x = (PetscReal)(start[X] + i % shw);
      auto g_y = (PetscReal)(start[Y] + (i / shw) % shw);
      auto g_z = (PetscReal)(start[Z] + (i / shw) / shw);
      cache[i] = sfunc(p_r[X] - g_x) * sfunc(p_r[Y] - g_y) * sfunc(p_r[Z] - g_z);
    }
  }
};

/// @todo Maybe it would be better to inline the calculation of
/// continuity equation into charge gathering so that in each cell
/// we will (1) substract the previous density, (2) calculate divergence.
PetscErrorCode ParticlesChargeDensity::collect()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(local_, 0.0));
  PetscCall(VecSet(field_, 0.0));

  PetscReal*** arr;
  PetscCall(DMDAVecGetArrayWrite(da_, local_, &arr));

  Shape shape;
  const PetscReal q = particles_.parameters.q;

#pragma omp parallel for private(shape)
  for (auto&& cell : particles_.storage) {
    for (auto&& point : cell) {
      shape.setup(point.r);

      for (PetscInt i = 0; i < shape.shm; ++i) {
        PetscInt g_x = shape.start[X] + i % shape.shw;
        PetscInt g_y = shape.start[Y] + (i / shape.shw) % shape.shw;
        PetscInt g_z = shape.start[Z] + (i / shape.shw) / shape.shw;

#pragma omp atomic update
        arr[g_z][g_y][g_x] += q * shape.cache[i] * particles_.n_Np(point);
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayWrite(da_, local_, &arr));
  PetscCall(DMLocalToGlobal(da_, local_, ADD_VALUES, field_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


ChargeConservation::ChargeConservation(DM da, std::vector<Vec> current_densities,
  std::vector<const interfaces::Particles*> particles)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/charge_conservation.txt"),
    current_densities(current_densities)
{
  PetscCheckAbort(current_densities.size() == particles.size() + 1, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Number of `current_densities` should be one more than `particles` to evaluate the total difference");

  for (const auto& sort : particles) {
    charge_densities.emplace_back(
      std::make_unique<ParticlesChargeDensity>(*sort));
  }

  Divergence divergence(da);
  PetscCallAbort(PETSC_COMM_WORLD, divergence.create_negative(&divE));
}

PetscErrorCode ChargeConservation::initialize()
{
  PetscFunctionBeginUser;
  for (auto& charge_density : charge_densities)
    charge_density->collect();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ChargeConservation::add_titles()
{
  PetscFunctionBeginUser;
  add_title("time");

  for (const auto& rho : charge_densities) {
    const auto& name = rho->particles_.parameters.sort_name;
    add_title("N1δQ_" + name);
    add_title("N2δQ_" + name);
  }

  add_title("Norm1(δQ)");
  add_title("Norm2(δQ)");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ChargeConservation::add_args(PetscInt t)
{
  PetscFunctionBeginUser;
  add_arg(t);

  // It is important to get `da` from `DistributionMoment` as it is reduced in dof
  DM da = charge_densities[0]->da_;

  Vec sum, diff;
  PetscReal norm[2];
  PetscCall(DMGetGlobalVector(da, &diff));
  PetscCall(DMGetGlobalVector(da, &sum));
  PetscCall(VecSet(sum, 0.0));

  PetscInt i = 0;
  for (; i < (PetscInt)charge_densities.size(); ++i) {
    // Computing partial derivative in time of charge density
    const auto& rho = charge_densities[i];
    PetscCall(VecCopy(rho->field_, diff));
    PetscCall(rho->collect());
    PetscCall(VecAYPX(diff, -1.0, rho->field_));
    PetscCall(VecScale(diff, 1.0 / dt));

    PetscCall(VecAXPY(sum, 1.0, diff));

    // Evaluating continuity equation
    const auto& currJe = current_densities[i];
    PetscCall(MatMultAdd(divE, currJe, diff, diff));
    PetscCall(VecNorm(diff, NORM_1_AND_2, norm));
    add_arg(norm[0]);
    add_arg(norm[1]);
  }

  PetscCall(DMRestoreGlobalVector(da, &diff));

  const auto& currJe = current_densities[i];
  PetscCall(MatMultAdd(divE, currJe, sum, sum));
  PetscCall(VecNorm(sum, NORM_1_AND_2, norm));
  add_arg(norm[0]);
  add_arg(norm[1]);

  PetscCall(DMRestoreGlobalVector(da, &sum));
  PetscFunctionReturn(PETSC_SUCCESS);
}
