#include "simulation.h"

#include "src/commands/builders/command_builder.h"
#include "src/diagnostics/builders/diagnostic_builder.h"
#include "src/diagnostics/energy_conservation.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"


namespace ecsim {

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;

  assembly_map.resize(world.size.elements_product());

  PetscCall(init_vectors());
  PetscCall(init_matrices());
  PetscCall(init_ksp_solvers());

  /// @todo The problem with simulation setup is growing, should be moved into interfaces!
  PetscCall(init_particles());

  std::vector<Command_up> presets;
  PetscCall(build_commands(*this, "Presets", presets));
  PetscCall(build_commands(*this, "StepPresets", step_presets_));

  LOG("Executing presets");
  for (auto&& preset : presets)
    preset->execute(start);

  if (!CONFIG().is_loaded_from_backup())
    PetscCall(VecAXPY(B, 1.0, B0));

  PetscCall(build_diagnostics(*this, diagnostics_));

  std::vector<const interfaces::Particles*> sorts;
  for (const auto& sort : particles_) {
    sorts.emplace_back(sort.get());
  }

  auto&& f_diag = std::make_unique<FieldsEnergy>(world.da, E, B);
  auto&& p_diag = std::make_unique<ParticlesEnergy>(sorts);

  diagnostics_.emplace_back(std::make_unique<EnergyConservation>(
    *this, std::move(f_diag), std::move(p_diag)));
  /// @todo all of the in-between code

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @todo Map one-to-one Appendix D of the article https://doi.org/10.1016/j.jcp.2017.01.002
PetscErrorCode Simulation::timestep_implementation(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  PetscCall(clear_sources());
  PetscCall(first_push());
  PetscCall(predict_fields());
  PetscCall(second_push());
  PetscCall(final_update());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(currI, 0.0));
  PetscCall(MatZeroEntries(matL));

  for (auto& sort : particles_)
    PetscCall(sort->clear_sources());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::first_push()
{
  PetscFunctionBeginUser;
  for (auto& sort : particles_)
    PetscCall(sort->first_push());

  PetscCall(update_cells_with_assembly());
  PetscCall(fill_ecsim_current());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::predict_fields()
{
  PetscFunctionBeginUser;
  // Storing `matL` to reuse it for ECSIM current calculation later
  Mat matA;
  PetscCall(MatDuplicate(matL, MAT_COPY_VALUES, &matA));  // matA = matL
  PetscCall(MatAYPX(matA, 2.0 * dt, matM, DIFFERENT_NONZERO_PATTERN));  // matA = matM + (2 * dt) * matA

  // Note that we use `matM` to construct the preconditioning matrix
  PetscCall(KSPSetOperators(ksp, matA, matM));
  PetscCall(KSPSetUp(ksp));

  PetscCall(advance_fields(ksp, currI, Eh));
  PetscCall(MatDestroy(&matA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::second_push()
{
  PetscFunctionBeginUser;
  DM da = world.da;
  PetscCall(DMGlobalToLocal(da, Eh, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, B, INSERT_VALUES, local_B));

  Vector3R*** arr_E;
  Vector3R*** arr_B;
  PetscCall(DMDAVecGetArrayRead(da, local_E, &arr_E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &arr_B));

  for (auto& sort : particles_) {
    sort->E = arr_E;
    sort->B = arr_B;
    PetscCall(sort->second_push());
    PetscCall(sort->correct_coordinates());
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &arr_E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &arr_B));

  PetscCall(update_cells_with_assembly());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::final_update()
{
  PetscFunctionBeginUser;
  Vec util;
  PetscReal norm;
  PetscCall(DMGetGlobalVector(world.da, &util));

  PetscCall(VecSet(util, 0.0));  // util = 0.0
  PetscCall(VecAXPBYPCZ(util, 2, -1, 1, Eh, E));  // E^{n+1} = 2 * E^{n+1/2} - E^{n}
  PetscCall(VecNorm(util, NORM_2, &norm));
  LOG("  Norm of the difference in electric fields between steps: {:.7f}", norm);

  PetscCall(VecSwap(util, E));
  PetscCall(DMRestoreGlobalVector(world.da, &util));

  PetscCall(MatMultAdd(rotE, Eh, B, B));  // B^{n+1} -= dt * rot(E^{n+1/2})
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::advance_fields(KSP ksp, Vec curr, Vec out)
{
  PetscFunctionBeginUser;
  Vec rhs;
  PetscCall(DMGetGlobalVector(world.da, &rhs));
  PetscCall(VecAXPY(B, -1.0, B0));

  PetscCall(VecCopy(curr, rhs));  // rhs = curr
  PetscCall(VecAXPBY(rhs, 2.0, -dt, E));  // rhs = 2 * E^{n} - (dt * rhs)
  PetscCall(MatMultAdd(rotB, B, rhs, rhs));  // rhs = rhs + dt * rotB(B^{n})

  PetscCall(KSPSolve(ksp, rhs, out));
  PetscCall(KSPGetSolution(ksp, &out));

  PetscCall(VecAXPY(B, +1.0, B0));
  PetscCall(DMRestoreGlobalVector(world.da, &rhs));

  // Convergence analysis
  LOG("  KSPSolve() has finished, KSPConvergedReasonView():");
  PetscCall(KSPConvergedReasonView(ksp, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::update_cells_with_assembly()
{
  PetscFunctionBeginUser;
  for (auto& sort : particles_)
    sort->update_cells();

  for (const auto& sort : particles_) {
    for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
      if (assembly_radius < 0) {
        assembly_map[g] = false;
        indices_assembled = false;
      }

      if (sort->storage[g].empty() || (indices_assembled && assembly_map[g]))
        continue;

      if (indices_assembled) {
        LOG("  Indices assembly has been broken by \"{}\"", sort->parameters.sort_name);
      }
      indices_assembled = false;

      if (assembly_radius <= 0) {
        assembly_map[g] = true;
        continue;
      }

      Vector3I vg{
        world.start[X] + g % world.size[X],
        world.start[Y] + (g / world.size[X]) % world.size[Y],
        world.start[Z] + (g / world.size[X]) / world.size[Y],
      };

      for (PetscInt i = 0; i < POW3(assembly_width); ++i) {
        Vector3I vgi{
          vg[X] - assembly_radius + i % assembly_width,
          vg[Y] - assembly_radius + (i / assembly_width) % assembly_width,
          vg[Z] - assembly_radius + (i / assembly_width) / assembly_width,
        };

        if (!is_point_within_bounds(vgi, world.start, world.size))
          continue;

        PetscInt j = world.s_g(  //
          vgi[X] - world.start[X],  //
          vgi[Y] - world.start[Y],  //
          vgi[Z] - world.start[Z]);

        assembly_map[j] = true;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::fill_ecsim_current()
{
  PetscFunctionBeginUser;
  PetscInt size = 0;
  get_array_offset(0, world.size.elements_product(), size);

  // Because matrix setup is collective, we must call it on all mpi ranks
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &indices_assembled, 1, MPIU_BOOL, MPI_BAND, PETSC_COMM_WORLD));

  if (!indices_assembled) {
    std::vector<PetscInt> coo_i(size, PETSC_DEFAULT);
    std::vector<PetscInt> coo_j(size, PETSC_DEFAULT);

    PetscReal ind_mem, val_mem;
    ind_mem = (PetscReal)size * 2 * sizeof(PetscInt) / 1e9;
    val_mem = (PetscReal)size * sizeof(PetscReal) / 1e9;
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &ind_mem, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &val_mem, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

    LOG("  Indices assembly was broken, recreating them");
    LOG("  Additional space for indices: {:4.3f} GB, values: {:4.3f} GB", ind_mem, val_mem);

    PetscCall(fill_matrix_indices(coo_i.data(), coo_j.data()));
    PetscCall(MatSetPreallocationCOOLocal(matL, size, coo_i.data(), coo_j.data()));
    indices_assembled = true;
  }

  std::vector<PetscReal> coo_v(size, 0.0);
  PetscCall(fill_ecsim_current(coo_v.data()));

  PetscCall(MatSetValuesCOO(matL, coo_v.data(), ADD_VALUES));
  PetscCall(MatScale(matL, 0.25 * dt));  // matL *= dt / 4
  PetscFunctionReturn(PETSC_SUCCESS);
}

constexpr PetscInt ind(PetscInt g, PetscInt c1, PetscInt c2)
{
  return g * POW2(3) + (c1 * 3 + c2);
}

PetscErrorCode Simulation::fill_matrix_indices(PetscInt* coo_i, PetscInt* coo_j)
{
  PetscFunctionBeginUser;
  constexpr PetscInt shr = 1;
  constexpr PetscInt shw = 2 * shr + 1;
  constexpr PetscInt m = POW3(shw);

  const PetscInt dims[4]{REP3_AP(world.gsize), 3};
  const PetscInt start[4]{REP3_AP(world.gstart), 0};

  auto remap_stencil = [&](const MatStencil& s) {
    auto in = (PetscInt*)&s;
    PetscInt tmp = *in++ - start[0];

    for (PetscInt j = 0; j < 3; ++j)
      if ((*in++ - start[j + 1]) < 0 || tmp < 0)
        tmp = -1;
      else
        tmp = tmp * dims[j + 1] + *(in - 1) - start[j + 1];

    return tmp;
  };

  PetscInt prev_g = 0;
  PetscInt off = 0;

#pragma omp parallel for firstprivate(prev_g, off)
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    if (!assembly_map[g])
      continue;

    get_array_offset(prev_g, g, off);
    prev_g = g;

    PetscInt* coo_ci = coo_i + off;
    PetscInt* coo_cj = coo_j + off;

    Vector3I vg{
      world.start[X] + g % world.size[X],
      world.start[Y] + (g / world.size[X]) % world.size[Y],
      world.start[Z] + (g / world.size[X]) / world.size[Y],
    };

    for (PetscInt g1 = 0; g1 < m; ++g1) {
      Vector3I vg1{
        vg[X] + g1 % shw - shr,
        vg[Y] + (g1 / shw) % shw - shr,
        vg[Z] + (g1 / shw) / shw - shr,
      };

      for (PetscInt g2 = 0; g2 < m; ++g2) {
        PetscInt gg = g1 * m + g2;

        Vector3I vg2{
          vg[X] + g2 % shw - shr,
          vg[Y] + (g2 / shw) % shw - shr,
          vg[Z] + (g2 / shw) / shw - shr,
        };

        for (PetscInt c = 0; c < Vector3I::dim; ++c) {
          coo_ci[ind(gg, X, c)] = remap_stencil(MatStencil{REP3_AP(vg1), X});
          coo_ci[ind(gg, Y, c)] = remap_stencil(MatStencil{REP3_AP(vg1), Y});
          coo_ci[ind(gg, Z, c)] = remap_stencil(MatStencil{REP3_AP(vg1), Z});

          coo_cj[ind(gg, c, X)] = remap_stencil(MatStencil{REP3_AP(vg2), X});
          coo_cj[ind(gg, c, Y)] = remap_stencil(MatStencil{REP3_AP(vg2), Y});
          coo_cj[ind(gg, c, Z)] = remap_stencil(MatStencil{REP3_AP(vg2), Z});
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::fill_ecsim_current(PetscReal* coo_v)
{
  PetscFunctionBeginUser;
  Vector3R*** arr_B;

  DM da = world.da;
  PetscCall(DMGlobalToLocal(da, B, INSERT_VALUES, local_B));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &arr_B));

  for (auto& sort : particles_) {
    sort->B = arr_B;
    PetscCall(DMDAVecGetArrayWrite(da, sort->local_currI, &sort->currI));
  }

  for (const auto& sort : particles_) {
    PetscInt prev_g = 0;
    PetscInt off = 0;

#pragma omp parallel for firstprivate(prev_g, off)
    for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
      if (sort->storage[g].empty())
        continue;

      get_array_offset(prev_g, g, off);
      prev_g = g;

      PetscReal* coo_cv = coo_v + off;
      sort->fill_ecsim_current(g, coo_cv);
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &arr_B));

  for (auto& sort : particles_) {
    PetscCall(DMDAVecRestoreArrayWrite(da, sort->local_currI, &sort->currI));
    PetscCall(DMLocalToGlobal(da, sort->local_currI, ADD_VALUES, sort->global_currI));
    PetscCall(VecAXPY(currI, 1.0, sort->global_currI));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

void Simulation::get_array_offset(PetscInt begin_g, PetscInt end_g, PetscInt& off)
{
  constexpr PetscInt shs = POW2(3 * POW3(3));
  for (PetscInt g = begin_g; g < end_g; ++g)
    if (assembly_map[g])
      off += shs;
}

PetscErrorCode Simulation::init_vectors()
{
  PetscFunctionBeginUser;
  DM da = world.da;
  PetscCall(DMCreateGlobalVector(da, &E));
  PetscCall(DMCreateGlobalVector(da, &Eh));
  PetscCall(DMCreateGlobalVector(da, &B));
  PetscCall(DMCreateGlobalVector(da, &B0));
  PetscCall(DMCreateGlobalVector(da, &currI));

  PetscCall(DMCreateLocalVector(da, &local_E));
  PetscCall(DMCreateLocalVector(da, &local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_matrices()
{
  PetscFunctionBeginUser;
  DM da = world.da;
  PetscCall(DMSetMatrixPreallocateOnly(da, PETSC_FALSE));
  PetscCall(DMSetMatrixPreallocateSkip(da, PETSC_TRUE));

  PetscCall(DMCreateMatrix(da, &matL));
  PetscCall(MatSetOption(matL, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));

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

PetscErrorCode Simulation::init_ksp_solvers()
{
  PetscFunctionBeginUser;
  static constexpr PetscReal atol = 1e-16;
  static constexpr PetscReal rtol = 1e-16;

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE));
  PetscCall(KSPSetReusePreconditioner(ksp, PETSC_TRUE));
  PetscCall(KSPSetTolerances(ksp, rtol, atol, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_particles()
{
  PetscFunctionBeginUser;
  LOG("Configuring particles");

  const Configuration::json_t& json = CONFIG().json;
  auto it = json.find("Particles");

  if (it == json.end() || it->empty())
    PetscFunctionReturn(PETSC_SUCCESS);

  for (auto&& info : *it) {
    SortParameters parameters;
    info.at("sort_name").get_to(parameters.sort_name);
    info.at("Np").get_to(parameters.Np);
    info.at("n").get_to(parameters.n);
    info.at("q").get_to(parameters.q);
    info.at("m").get_to(parameters.m);

    if (info.contains("T")) {
      PetscReal T;
      info.at("T").get_to(T);
      parameters.Tx = T;
      parameters.Ty = T;
      parameters.Tz = T;
    }
    else {
      info.at("Tx").get_to(parameters.Tx);
      info.at("Ty").get_to(parameters.Ty);
      info.at("Tz").get_to(parameters.Tz);
    }

    particles_.emplace_back(std::make_unique<Particles>(*this, parameters));

    PetscReal T = std::hypot(parameters.Tx, parameters.Ty, parameters.Tz);
    PetscReal V = std::sqrt(T / (parameters.m * 511.0));
    PetscReal H = std::hypot(Dx[X] / V, Dx[Y] / V, Dx[Z] / V);

    LOG("  {} are added:", parameters.sort_name);
    LOG("    temperature,         T = {:.3e} [KeV]", T);
    LOG("    thermal velocity, v_th = {:.3e} [c]", V);
    LOG("    cell-heating, Dx / L_d = {:.3e} [unit]", H);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

Simulation::~Simulation()
{
  PetscFunctionBeginUser;
  PetscCallVoid(KSPDestroy(&ksp));

  PetscCallVoid(MatDestroy(&matL));
  PetscCallVoid(MatDestroy(&matM));
  PetscCallVoid(MatDestroy(&rotE));
  PetscCallVoid(MatDestroy(&rotB));

  PetscCallVoid(VecDestroy(&E));
  PetscCallVoid(VecDestroy(&Eh));
  PetscCallVoid(VecDestroy(&B));
  PetscCallVoid(VecDestroy(&B0));
  PetscCallVoid(VecDestroy(&currI));

  PetscCallVoid(VecDestroy(&local_E));
  PetscCallVoid(VecDestroy(&local_B));
  PetscFunctionReturnVoid();
}


Vec Simulation::get_named_vector(std::string_view name)
{
  if (name == "E")
    return E;
  if (name == "B")
    return B;
  if (name == "B0")
    return B0;
  if (name == "J")
    return currI;
  throw std::runtime_error("Unknown vector name " + std::string(name));
}

Particles& Simulation::get_named_particles(std::string_view name)
{
  return interfaces::Simulation::get_named_particles(name, particles_);
}

Simulation::NamedValues<Vec> Simulation::get_backup_fields()
{
  return NamedValues<Vec>{{"E", E}, {"B", B}, {"B0", B0}};
}

Simulation::NamedValues<interfaces::Particles*> Simulation::get_backup_particles()
{
  NamedValues<interfaces::Particles*> particles;
  for (auto&& sort : particles_)
    particles.insert(std::make_pair(sort->parameters.sort_name, sort.get()));
  return particles;
}

}  // namespace ecsim
