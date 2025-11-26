#include "simulation.h"

#include "src/diagnostics/energy_conservation.h"
#include "src/diagnostics/momentum_conservation.h"
#include "src/utils/geometries.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"


namespace ecsim {

static constexpr PetscReal atol = 1e-7;
static constexpr PetscReal rtol = 1e-7;

Vector3R interpolate_E_s1(Arr E_g, const Vector3R& r)
{
  Vector3R E_p;

  /// @todo Create some weights calculator
  PetscInt ixn, iyn, izn, ixs, iys, izs;
  PetscReal xn, yn, zn, xs, ys, zs;
  PetscReal wnx[2], wny[2], wnz[2], wsx[2], wsy[2], wsz[2];

  xn = r[X] / dx;
  yn = r[Y] / dy;
  zn = r[Z] / dz;
  xs = xn - 0.5;
  ys = yn - 0.5;
  zs = zn - 0.5;

  ixn = (PetscInt)floor(xn);
  iyn = (PetscInt)floor(yn);
  izn = (PetscInt)floor(zn);
  ixs = (PetscInt)floor(xs);
  iys = (PetscInt)floor(ys);
  izs = (PetscInt)floor(zs);

  wnx[1] = (xn - ixn);
  wny[1] = (yn - iyn);
  wnz[1] = (zn - izn);
  wnx[0] = 1 - wnx[1];
  wny[0] = 1 - wny[1];
  wnz[0] = 1 - wnz[1];

  wsx[1] = (xs - ixs);
  wsy[1] = (ys - iys);
  wsz[1] = (zs - izs);
  wsx[0] = 1 - wsx[1];
  wsy[0] = 1 - wsy[1];
  wsz[0] = 1 - wsz[1];

  PetscInt i, j, k;
  PetscReal s[3];

  for (k = 0; k < 2; ++k) {
    for (j = 0; j < 2; ++j) {
      for (i = 0; i < 2; ++i) {
        s[X] = wnz[k] * wny[j] * wsx[i];
        s[Y] = wnz[k] * wsy[j] * wnx[i];
        s[Z] = wsz[k] * wny[j] * wnx[i];

        E_p[X] += E_g[izn + k][iyn + j][ixs + i][X] * s[X];
        E_p[Y] += E_g[izn + k][iys + j][ixn + i][Y] * s[Y];
        E_p[Z] += E_g[izs + k][iyn + j][ixn + i][Z] * s[Z];
      }
    }
  }
  return E_p;
}

Vector3R interpolate_B_s1(Arr B_g, const Vector3R& r)
{
  Vector3R B_p;

  /// @todo Create some weights calculator
  PetscInt ixn, iyn, izn, ixs, iys, izs;
  PetscReal xn, yn, zn, xs, ys, zs;
  PetscReal wnx[2], wny[2], wnz[2], wsx[2], wsy[2], wsz[2];

  xn = r[X] / dx;
  yn = r[Y] / dy;
  zn = r[Z] / dz;
  xs = xn - 0.5;
  ys = yn - 0.5;
  zs = zn - 0.5;

  ixn = (PetscInt)floor(xn);
  iyn = (PetscInt)floor(yn);
  izn = (PetscInt)floor(zn);
  ixs = (PetscInt)floor(xs);
  iys = (PetscInt)floor(ys);
  izs = (PetscInt)floor(zs);

  wnx[1] = (xn - ixn);
  wny[1] = (yn - iyn);
  wnz[1] = (zn - izn);
  wnx[0] = 1 - wnx[1];
  wny[0] = 1 - wny[1];
  wnz[0] = 1 - wnz[1];

  wsx[1] = (xs - ixs);
  wsy[1] = (ys - iys);
  wsz[1] = (zs - izs);
  wsx[0] = 1 - wsx[1];
  wsy[0] = 1 - wsy[1];
  wsz[0] = 1 - wsz[1];

  PetscInt i, j, k;
  PetscReal s[3];

  for (k = 0; k < 2; ++k) {
    for (j = 0; j < 2; ++j) {
      for (i = 0; i < 2; ++i) {
        s[X] = wsz[k] * wsy[j] * wnx[i];
        s[Y] = wsz[k] * wny[j] * wsx[i];
        s[Z] = wnz[k] * wsy[j] * wsx[i];

        B_p[X] += B_g[izs + k][iys + j][ixn + i][X] * s[X];
        B_p[Y] += B_g[izs + k][iyn + j][ixs + i][Y] * s[Y];
        B_p[Z] += B_g[izn + k][iys + j][ixs + i][Z] * s[Z];
      }
    }
  }
  return B_p;
}

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  SyncClock init_clock;
  PetscCall(init_clock.push(__FUNCTION__));
  PetscCall(init_log_stages());
  PetscCall(PetscLogStagePush(stagenums[0]));

  assembly_map.resize(world.size.elements_product());

  PetscCall(init_vectors());
  PetscCall(init_matrices());
  PetscCall(init_ksp_solvers());
  PetscCall(init_particles());

  std::vector<const interfaces::Particles*> sorts;
  for (const auto& sort : particles_) {
    sorts.emplace_back(sort.get());
  }

  diagnostics_.emplace_back(
    std::make_unique<MomentumConservation>(world.da, E, sorts));

  auto&& e_diag = std::make_unique<Energy>(E, B, sorts);

  diagnostics_.emplace_back(
    std::make_unique<EnergyConservation>(*this, std::move(e_diag)));

  PetscCall(PetscLogStagePop());
  PetscCall(init_clock.pop());
  LOG("Initialization of ecsim took {:6.4e} seconds", init_clock.get(__FUNCTION__));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::timestep_implementation(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(clear_sources());
  PetscCall(first_push());
  PetscCall(advance_fields(matL));
  PetscCall(second_push());
  PetscCall(final_update());
  PetscCall(clock.log_timings());

  if (t % diagnose_period == 0) {
    for (auto& sort : particles_)
      PetscCall(sort->log_distribution());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push("clear_sources1"));
  PetscCall(PetscLogStagePush(stagenums[1]));

  PetscCall(VecSet(currI, 0));
  PetscCall(MatZeroEntries(matL));

  for (auto& sort : particles_)
    PetscCall(sort->clear_sources());

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::first_push()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));
  PetscCall(PetscLogStagePush(stagenums[2]));

  for (auto& sort : particles_)
    PetscCall(sort->first_push());

  PetscCall(update_cells_with_assembly());
  PetscCall(fill_ecsim_current());

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::advance_fields(Mat matA)
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));
  PetscCall(PetscLogStagePush(stagenums[3]));

  PetscCall(MatAXPY(matA, 1, matM, DIFFERENT_NONZERO_PATTERN));  // matA += matM

  // Note that we use `matM` to construct the preconditioning matrix
  PetscCall(KSPSetOperators(ksp, matA, matM));
  PetscCall(KSPSetUp(ksp));

  PetscCall(advance_fields(ksp, currI, Ep));

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::second_push()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));
  PetscCall(PetscLogStagePush(stagenums[4]));

  DM da = world.da;
  PetscCall(DMGlobalToLocal(da, Ep, INSERT_VALUES, local_E));
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

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::final_update()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push("final_update1"));
  PetscCall(PetscLogStagePush(stagenums[5]));

  Vec util;
  PetscReal norm;
  PetscCall(DMGetGlobalVector(world.da, &util));

  PetscCall(VecSet(util, 0));
  PetscCall(VecAXPBYPCZ(util, 2, -1, 1, Ep, E));  // E^{n+1} = 2 * E^{n+1/2} - E^{n}
  PetscCall(VecNorm(util, NORM_2, &norm));
  LOG("  Norm of the difference in electric fields between steps: {:.7f}", norm);

  PetscCall(VecSwap(util, E));
  PetscCall(DMRestoreGlobalVector(world.da, &util));

  PetscCall(MatMultAdd(rotE, Ep, B, B));  // B^{n+1} -= dt * rot(E^{n+1/2})

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::advance_fields(KSP ksp, Vec curr, Vec out)
{
  PetscFunctionBeginUser;
  Vec rhs;
  PetscCall(DMGetGlobalVector(world.da, &rhs));
  PetscCall(VecAXPY(B, -1.0, B0));

  PetscCall(VecCopy(curr, rhs));
  PetscCall(VecAXPBY(rhs, 2.0, -dt, E));  // rhs = 2 * E^{n} - dt * rhs
  PetscCall(MatMultAdd(rotB, B, rhs, rhs));  // rhs = rhs + dt * rotB(B^{n})

  PetscCall(KSPSolve(ksp, rhs, out));
  PetscCall(KSPGetSolution(ksp, &out));

  PetscCall(VecAXPY(B, +1.0, B0));
  PetscCall(DMRestoreGlobalVector(world.da, &rhs));

  // Convergence analysis
  const char* name;
  PetscCall(PetscObjectGetName((PetscObject)ksp, &name));
  LOG("  KSPSolve() has finished for \"{}\", KSPConvergedReasonView():", name);
  PetscCall(KSPConvergedReasonView(ksp, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @note Since we use one global Lapenta matrix, test on `indices_assembled` should include particles of all sorts.
/// @note This routine _must_ be called before `fill_matrix_indices()`
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
  PetscFunctionReturn(PETSC_SUCCESS);
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

    const Vector3I vg{
      world.start[X] + g % world.size[X],
      world.start[Y] + (g / world.size[X]) % world.size[Y],
      world.start[Z] + (g / world.size[X]) / world.size[Y],
    };

    PetscInt c1, si1, sj1, sk1, oi1, oj1, ok1, i1, j1, k1;
    PetscInt c2, si2, sj2, sk2, oi2, oj2, ok2, i2, j2, k2;
    PetscInt i, j, ind;

    for (c1 = 0; c1 < 3; ++c1) {
      si1 = (c1 == 0 ? 3 : 2);
      sj1 = (c1 == 1 ? 3 : 2);
      sk1 = (c1 == 2 ? 3 : 2);

      oi1 = (c1 == 0 ? -1 : 0);
      oj1 = (c1 == 1 ? -1 : 0);
      ok1 = (c1 == 2 ? -1 : 0);

      for (k1 = 0; k1 < sk1; ++k1) {
        for (j1 = 0; j1 < sj1; ++j1) {
          for (i1 = 0; i1 < si1; ++i1) {
            const Vector3I vg1{
              vg[X] + i1 + oi1,
              vg[Y] + j1 + oj1,
              vg[Z] + k1 + ok1,
            };

            for (c2 = 0; c2 < 3; ++c2) {
              si2 = (c2 == 0 ? 3 : 2);
              sj2 = (c2 == 1 ? 3 : 2);
              sk2 = (c2 == 2 ? 3 : 2);

              oi2 = (c2 == 0 ? -1 : 0);
              oj2 = (c2 == 1 ? -1 : 0);
              ok2 = (c2 == 2 ? -1 : 0);

              for (k2 = 0; k2 < sk2; ++k2) {
                for (j2 = 0; j2 < sj2; ++j2) {
                  for (i2 = 0; i2 < si2; ++i2) {
                    const Vector3I vg2{
                      vg[X] + i2 + oi2,
                      vg[Y] + j2 + oj2,
                      vg[Z] + k2 + ok2,
                    };

                    i = (k1 * sj1 + j1) * si1 + i1;
                    j = (k2 * sj2 + j2) * si2 + i2;
                    ind = (c1 * 3 + c2) * POW2(12) + (i * 12 + j);

                    coo_ci[ind] = remap_stencil(MatStencil{REP3_AP(vg1), c1});
                    coo_cj[ind] = remap_stencil(MatStencil{REP3_AP(vg2), c2});
                  }  // G'x
                }  // G'y
              }  // G'z
            }
          }  // Gx
        }  // Gy
      }  // Gz
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
    PetscCall(sort->fill_ecsim_current(coo_v));
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &arr_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

void Simulation::get_array_offset(PetscInt begin_g, PetscInt end_g, PetscInt& off)
{
  constexpr PetscInt shs = POW2(3 * 2 * 2 * 3);
  for (PetscInt g = begin_g; g < end_g; ++g)
    if (assembly_map[g])
      off += shs;
}


PetscErrorCode Simulation::init_log_stages()
{
  PetscFunctionBeginUser;
  PetscCall(PetscClassIdRegister("ecsimcorr::Simulation", &classid));
  PetscCall(PetscLogEventRegister("fill_ecsim_curr", classid, &events[0]));

  PetscCall(PetscLogStageRegister("Initialization", &stagenums[0]));
  PetscCall(PetscLogStageRegister("Clear sources", &stagenums[1]));
  PetscCall(PetscLogStageRegister("First push", &stagenums[2]));
  PetscCall(PetscLogStageRegister("Advance field", &stagenums[3]));
  PetscCall(PetscLogStageRegister("Second push", &stagenums[4]));
  PetscCall(PetscLogStageRegister("Final update", &stagenums[5]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_particles()
{
  PetscFunctionBeginUser;
  PetscCall(interfaces::Simulation::init_particles(*this, particles_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_vectors()
{
  PetscFunctionBeginUser;
  DM da = world.da;
  PetscCall(DMCreateGlobalVector(da, &E));
  PetscCall(DMCreateGlobalVector(da, &Ep));
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
  PetscCall(MatProductNumeric(matM));

  PetscCall(MatScale(matM, 0.5 * dt * dt));
  PetscCall(MatShift(matM, 2.0));

  PetscCall(MatScale(rotE, -dt));
  PetscCall(MatScale(rotB, +dt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_ksp_solvers()
{
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE));
  PetscCall(KSPSetReusePreconditioner(ksp, PETSC_TRUE));
  PetscCall(KSPSetTolerances(ksp, rtol, atol, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(interfaces::Simulation::finalize());

  PetscCall(KSPDestroy(&ksp));

  PetscCall(MatDestroy(&matL));
  PetscCall(MatDestroy(&matM));
  PetscCall(MatDestroy(&rotE));
  PetscCall(MatDestroy(&rotB));

  PetscCall(VecDestroy(&E));
  PetscCall(VecDestroy(&Ep));
  PetscCall(VecDestroy(&B));
  PetscCall(VecDestroy(&B0));
  PetscCall(VecDestroy(&currI));

  PetscCall(VecDestroy(&local_E));
  PetscCall(VecDestroy(&local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}


Vec Simulation::get_named_vector(std::string_view name) const
{
  static const std::unordered_map<std::string_view, Vec> map{
    {"E", E},
    {"B", B},
    {"B0", B0},
    {"J", currI},
  };
  return map.at(name);
}

Simulation::NamedValues<Vec> Simulation::get_backup_fields() const
{
  return {{"E", E}, {"B", B}, {"B0", B0}};
}

}  // namespace ecsim
