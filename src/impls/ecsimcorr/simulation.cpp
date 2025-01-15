#include "simulation.h"

#include "src/commands/inject_particles.h"
#include "src/commands/remove_particles.h"
#include "src/commands/setup_magnetic_field.h"
#include "src/diagnostics/builders/diagnostic_builder.h"
#include "src/impls/ecsimcorr/energy.h"
#include "src/utils/operators.h"
#include "src/utils/particles_load.hpp"
#include "src/utils/random_generator.h"
#include "src/utils/utils.h"


namespace ecsimcorr {

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscCall(init_vectors());
  PetscCall(init_matrices());
  PetscCall(init_ksp_solvers());

  constexpr PetscInt Np = 1;

  auto build_particles = [&]() {
    const Configuration::json_t& particles_info = CONFIG().json.at("Particles");

    for (auto&& info : particles_info) {
      SortParameters parameters;
      info.at("sort_name").get_to(parameters.sort_name);
      parameters.Np = Np;
      info.at("n").get_to(parameters.n);
      info.at("q").get_to(parameters.q);
      info.at("m").get_to(parameters.m);
      info.at("T").get_to(parameters.Tx);
      info.at("T").get_to(parameters.Ty);
      info.at("T").get_to(parameters.Tz);
      particles_.emplace_back(std::make_unique<Particles>(*this, parameters));
    }
  };

  build_particles();


  for (PetscInt z = 0; z < geom_nz; ++z) {
    for (PetscInt y = 0; y < geom_ny; ++y) {
      for (PetscInt x = 0; x < geom_nx; ++x) {
        for (PetscInt n = 0; n < Np; ++n) {
          Vector3R r{
            (PetscReal)x + random_01() * dx,
            (PetscReal)y + random_01() * dy,
            (PetscReal)z + random_01() * dz,
          };

          for (auto&& sort : particles_) {
            MaxwellianMomentum momentum(sort->parameters(), true);
            sort->add_particle(Point(r, momentum(r)));
          }
        }
      }
    }
  }

  PetscCall(build_diagnostics(*this, diagnostics_));
  diagnostics_.emplace_back(std::make_unique<EnergyDiagnostic>(*this));

  PetscCall(init_log_stages());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_vectors()
{
  PetscFunctionBeginUser;
  DM da = world_.da;
  PetscCall(DMCreateGlobalVector(da, &E));
  PetscCall(DMCreateGlobalVector(da, &En));
  PetscCall(DMCreateGlobalVector(da, &B));
  PetscCall(DMCreateGlobalVector(da, &B0));
  PetscCall(DMCreateGlobalVector(da, &currI));
  PetscCall(DMCreateGlobalVector(da, &currJ));
  PetscCall(DMCreateGlobalVector(da, &currJe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_matrices()
{
  PetscFunctionBeginUser;
  DM da = world_.da;
  PetscCall(DMCreateMatrix(da, &matL));
  PetscCall(MatSetOption(matL, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
  PetscCall(MatDuplicate(matL, MAT_DO_NOT_COPY_VALUES, &matA));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rotE));
  PetscCall(rotor.create_negative(&rotB));

  PetscCall(MatProductCreate(rotB, rotE, nullptr, &matM));
  PetscCall(MatProductSetType(matM, MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(matM));
  PetscCall(MatProductSymbolic(matM));
  PetscCall(MatProductNumeric(matM));  // matM = rotB(rotE())
  PetscCall(MatScale(matM, 0.5 * POW2(dt)));  // matM = dt^2 / 2 * matM
  PetscCall(MatShift(matM, 2.0));  // matM = 2 * I + matM

  PetscCall(MatScale(rotE, -dt));
  PetscCall(MatScale(rotB, +dt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @todo We can optimize `correct` ksp further, by choosing appropriate KSPType
PetscErrorCode Simulation::init_ksp_solvers()
{
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &predict));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &correct));
  PetscCall(KSPSetErrorIfNotConverged(predict, PETSC_TRUE));
  PetscCall(KSPSetErrorIfNotConverged(correct, PETSC_TRUE));

  static constexpr PetscReal atol = 1e-10;
  static constexpr PetscReal rtol = 1e-10;
  PetscCall(KSPSetTolerances(predict, atol, rtol, PETSC_CURRENT, PETSC_CURRENT));
  PetscCall(KSPSetTolerances(correct, atol, rtol, PETSC_CURRENT, PETSC_CURRENT));

  PC pc;
  PetscCall(KSPGetPC(predict, &pc));

  // `matA` can be overwritten during `advance_fields()`, we would use only stored copy of `matL`
  PetscCall(PCFactorSetUseInPlace(pc, PETSC_TRUE));

  PetscCall(KSPSetOperators(correct, matM, matM));
  PetscCall(KSPSetUp(correct));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_log_stages()
{
  PetscFunctionBeginUser;
  PetscLogStageRegister("Clear sources", &stagenums[0]);
  PetscLogStageRegister("First particles push", &stagenums[1]);
  PetscLogStageRegister("Predict electric field", &stagenums[2]);
  PetscLogStageRegister("Second particles push", &stagenums[3]);
  PetscLogStageRegister("Correct electric and magnetic fields", &stagenums[4]);
  PetscLogStageRegister("Final particles update", &stagenums[5]);
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::timestep_implementation(timestep_t /* timestep */)
{
  PetscFunctionBeginUser;
  PetscLogStagePush(stagenums[0]);

  PetscCall(clear_sources());

  PetscLogStagePop();
  PetscLogStagePush(stagenums[1]);

  for (auto& sort : particles_)
    PetscCall(sort->first_push());

  PetscLogStagePop();
  PetscLogStagePush(stagenums[2]);

  PetscCall(predict_fields());

  PetscLogStagePop();
  PetscLogStagePush(stagenums[3]);

  for (auto& sort : particles_)
    PetscCall(sort->second_push());

  PetscLogStagePop();
  PetscLogStagePush(stagenums[4]);

  // PetscCall(correct_fields());

  PetscLogStagePop();
  PetscLogStagePush(stagenums[5]);

  for (auto& sort : particles_) {
    // PetscCall(sort->final_update());
    PetscCall(sort->correct_coordinates());

    /// @todo Testing petsc as a computational server first
    /// PetscCall(sort->communicate());
  }

  PetscLogStagePop();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::clear_sources()
{
  PetscFunctionBeginUser;
  w1 = 0.0;
  w2 = 0.0;
  PetscCall(VecSet(currI, 0.0));
  PetscCall(VecSet(currJ, 0.0));
  PetscCall(VecSet(currJe, 0.0));
  PetscCall(MatZeroEntries(matL));

  for (auto& sort : particles_)
    PetscCall(sort->clear_sources());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::predict_fields()
{
  PetscFunctionBeginUser;
  // Storing identity current `currI` before we use it
  // as a storage for the right hand side of the `ksp`
  PetscCall(VecCopy(currI, currJ));  // currJ = currI

  // The same copying is made for `matL`
  PetscCall(MatCopy(matL, matA, DIFFERENT_NONZERO_PATTERN));  // matA = matL
  PetscCall(MatAYPX(matA, 0.5 * POW2(dt), matM, DIFFERENT_NONZERO_PATTERN));  // matA = matM + dt^2 / 2 * matA

  PetscCall(KSPSetOperators(predict, matA, matA));
  PetscCall(KSPSetUp(predict));

  // Solving the Maxwell's equations to find prediction of E'^{n+1/2}
  PetscCall(advance_fields(predict, currI));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::correct_fields()
{
  PetscFunctionBeginUser;
  PetscCall(MatScale(matL, 0.5 * dt));  // matL = dt / 2 * matL
  PetscCall(MatMultAdd(matL, En, currJ, currJ));  // currJ = currJ + matL * E'^{n+1/2}

  PetscCall(VecDot(currJ, En, &w1));  // w1 = (currJ, E'^{n+1/2})
  LOG("  Work of the predicted field ((ECSIM cur.) * E_pred): {}", w1);
  DM da = world_.da;

  Vec diff;
  PetscReal norm;
  PetscCall(DMGetGlobalVector(da, &diff));
  PetscCall(VecWAXPY(diff, -1, currJ, currJe));
  PetscCall(VecNorm(diff, NORM_2, &norm));
  PetscCall(DMRestoreGlobalVector(da, &diff));
  LOG("  Norm of the difference in ECSIM and Esirkepov currents: {}", norm);

  PetscCall(VecCopy(currJe, currJ));  // currJ = currJe

  // Solving Maxwell's equation to find correct
  // E^{n+1/2}, satisfying continuity equation
  PetscCall(advance_fields(correct, currJe));

  PetscCall(VecDot(currJ, En, &w2));  // w2 = (currJ, E^{n+1/2})
  LOG("  Work of the corrected field ((Esirkepov cur.) * E_corr): {}", w2);

  Vec newE;
  PetscCall(DMGetGlobalVector(da, &newE));
  PetscCall(VecAXPBYPCZ(newE, 2, -1, 1, En, E));  // E^{n+1} = 2 * E^{n+1/2} - E^{n}
  PetscCall(VecNorm(newE, NORM_2, &norm));
  PetscCall(VecSwap(newE, E));
  PetscCall(DMRestoreGlobalVector(da, &newE));
  LOG("  Norm of the difference in predicted and corrected fields: {}", norm);

  PetscCall(MatMultAdd(rotE, En, B, B));  // B^{n+1} -= dt * rot(E^{n+1/2})
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::advance_fields(KSP ksp, Vec rhs)
{
  PetscFunctionBeginUser;
  PetscCall(VecAXPBY(rhs, 2.0, -dt, E));  // rhs = 2 * E^{n} - (dt * rhs)
  PetscCall(MatMultAdd(rotB, B, rhs, rhs));  // rhs = rhs + rotB(B^{n})

  PetscCall(KSPSolve(ksp, rhs, En));
  PetscCall(KSPGetSolution(ksp, &En));

  // Convergence analysis
  PetscCall(KSPConvergedReasonView(ksp, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

Vec Simulation::get_named_vector(std::string_view name) const
{
  if (name == "E^n")
    return E;
  if (name == "E^{n+1/2}")
    return En;
  if (name == "B^n")
    return B;
  if (name == "B^0")
    return B0;
  if (name == "I")
    return currI;
  if (name == "J_{ecsim}")
    return currI;
  if (name == "J_{esirkepov}")
    return currI;
  throw std::runtime_error("Unknown vector name " + std::string(name));
}

const Particles& Simulation::get_named_particles(std::string_view name) const
{
  auto it = std::find_if(particles_.begin(), particles_.end(),  //
    [&](const auto& sort) {
      return sort->parameters().sort_name == name;
    });

  if (it == particles_.end())
    throw std::runtime_error("No particles with name " + std::string(name));
  return **it;
}


Simulation::~Simulation()
{
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&E));
  PetscCallVoid(VecDestroy(&En));
  PetscCallVoid(VecDestroy(&B));
  PetscCallVoid(VecDestroy(&B0));
  PetscCallVoid(VecDestroy(&currI));
  PetscCallVoid(VecDestroy(&currJ));
  PetscCallVoid(VecDestroy(&currJe));

  PetscCallVoid(MatDestroy(&matL));
  PetscCallVoid(MatDestroy(&matA));
  PetscCallVoid(MatDestroy(&matM));
  PetscCallVoid(MatDestroy(&rotE));
  PetscCallVoid(MatDestroy(&rotB));

  PetscCallVoid(KSPDestroy(&predict));
  PetscCallVoid(KSPDestroy(&correct));
  PetscFunctionReturnVoid();
}

}  // namespace ecsimcorr
