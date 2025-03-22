#include "charge_conservation.h"

#include "src/utils/configuration.h"
#include "src/utils/operators.h"

ChargeConservation::ChargeConservation(DM da, std::vector<Vec> current_densities,
  std::vector<const interfaces::Particles*> particles)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/charge_conservation.txt"),
    current_densities(current_densities)
{
  PetscCheckAbort(current_densities.size() == particles.size() + 1, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Number of `current_densities` should be one more than `particles` to evaluate the total difference");

  FieldView::Region region{
    .dim = 3,
    .dof = 1,
    .start = Vector4I(0, 0, 0, 0),
    .size = Vector4I(geom_nx, geom_ny, geom_nz, 3),
  };

  for (const auto& sort : particles) {
    charge_densities.emplace_back(DistributionMoment::create(
      "", *sort, moment_from_string("Density"), region));
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
  for (const auto& rho : charge_densities) {
    const auto& name = rho->particles_.parameters.sort_name;
    add_title("N1δQ_" + name);
    add_title("N2δQ_" + name);
  }

  add_title("Norm1(δQ)");
  add_title("Norm2(δQ)");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ChargeConservation::add_args()
{
  PetscFunctionBeginUser;
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
    PetscCall(VecScale(diff, rho->particles_.parameters.q / dt));

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
