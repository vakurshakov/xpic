#include "src/impls/ecsimcorr/charge_conservation.h"

#include <iomanip>

#include "src/utils/configuration.h"
#include "src/utils/operators.h"
#include "src/utils/vector_utils.h"

namespace ecsimcorr {

ChargeConservation::ChargeConservation(const Simulation& simulation)
  : simulation(simulation)
{
  file_ = SyncFile(CONFIG().out_dir + "/charge_conservation.dat");

  FieldView::Region region{
    .dim = 3,
    .dof = 1,
    .start = vector_cast(Vector3I{0}),
    .size = vector_cast(Vector3I{Geom_n}),
  };

  for (const auto& particles : simulation.particles_) {
    charge_densities.emplace_back(DistributionMoment::create(
      "", *particles, moment_from_string("Density"), region));
  }

  Divergence divergence(simulation.world_.da);
  PetscCallVoid(divergence.create_negative(&divE));
}

PetscErrorCode ChargeConservation::diagnose(timestep_t t)
{
  PetscFunctionBeginUser;
  if (t == 0) {
    PetscCall(write_header());

    for (auto& charge_density : charge_densities)
      charge_density->collect();
  }

  auto output = [&](PetscReal x) {
    file_() << std::setw(14) << x;
  };

  DM da = charge_densities[0]->da_;

  Vec sum, diff;
  PetscReal norm[2];
  PetscCall(DMGetGlobalVector(da, &diff));
  PetscCall(DMGetGlobalVector(da, &sum));
  PetscCall(VecSet(sum, 0.0));

  for (auto& charge_density : charge_densities) {
    PetscCall(VecCopy(charge_density->field_, diff));

    charge_density->collect();

    PetscCall(VecAYPX(diff, -1.0, charge_density->field_));
    PetscCall(VecScale(diff, 1.0 / dt));
    PetscCall(VecScale(diff, charge_density->particles_.parameters().q));

    PetscCall(VecAXPY(sum, 1.0, diff));

    auto&& particles = dynamic_cast<const Particles&>(charge_density->particles_);
    PetscCall(MatMultAdd(divE, particles.global_currJe, diff, diff));
    PetscCall(VecNorm(diff, NORM_1_AND_2, norm));
    output(norm[0]);
    output(norm[1]);
  }

  PetscCall(DMRestoreGlobalVector(da, &diff));

  PetscCall(MatMultAdd(divE, simulation.currJe, sum, sum));
  PetscCall(VecNorm(sum, NORM_1_AND_2, norm));
  output(norm[0]);
  output(norm[1]);

  PetscCall(DMRestoreGlobalVector(da, &sum));

  file_() << "\n";

  if (t % diagnose_period == 0)
    file_.flush();

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ChargeConservation::write_header()
{
  PetscFunctionBeginUser;
  for (const auto& particles : simulation.particles_) {
    auto&& name = particles->parameters().sort_name;
    file_() << "Norm1(dC_" << name << ")\t";
    file_() << "Norm2(dC_" << name << ")\t";
  }

  file_() << "Norm1(dC)\t";
  file_() << "Norm2(dC)\t";
  file_() << "\n";
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace ecsimcorr
