#include "src/impls/eccapfim/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  " Warm periodic electrons (ions are a background) as in eccapfim_ex4,   \n"
  " timed by phases of a step. Options: -n (cells per axis) -ppc -steps.  \n"
  " Usage: [mpiexec -n N] ./eccapfim_warm_plasma_suite.out [snes options] \n"
  "                                                             [options] \n";

PetscErrorCode event_time(const char* name, PetscInt* count, PetscReal* time);

void overwrite_config(PetscInt n, PetscInt ppc, PetscInt steps);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(PetscLogDefaultBegin());

  PetscInt n = 16, ppc = 64, steps = 5;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ppc", &ppc, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-steps", &steps, NULL));

  overwrite_config(n, ppc, steps);

  eccapfim::Simulation simulation;
  PetscCall(simulation.initialize());

  PetscLogDouble t0, t1;
  PetscCall(PetscTime(&t0));
  PetscCall(simulation.calculate());
  PetscCall(PetscTime(&t1));

  const auto& sort = *simulation.particles_[0];
  PetscInt calls, size;
  PetscReal prepare, kernel, update_cells;
  PetscCall(event_time("prepare_storage", &calls, &prepare));
  PetscCall(event_time("form_iteration", &calls, &kernel));
  PetscCall(event_time("update_cells", &size, &update_cells));

  PetscLogStage stage;
  PetscEventPerfInfo solve;
  PetscCall(PetscLogStageGetId("Calc iteration", &stage));
  PetscCall(PetscLogStageGetPerfInfo(stage, &solve));

  const PetscReal arr[]{
    prepare,
    sort.get_interpolate_time(),
    sort.get_cn_push_time(),
    sort.get_decompose_time(),
    kernel -                         //
      sort.get_interpolate_time() -  //
      sort.get_cn_push_time() -      //
      sort.get_decompose_time(),
    kernel,
    solve.time - kernel,
    update_cells,
    t1 - t0,
  };
  PetscReal max[std::size(arr)];
  PetscCallMPI(MPI_Allreduce(arr, max, (PetscMPIInt)std::size(arr), MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));

  PetscMPIInt ranks;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
  PetscInt np = (PetscInt)n * n * n * ppc;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "n=%d ppc=%d (%d particles) steps=%d ranks=%d threads=%d, form_iteration calls per step %.1f\n",
    n, ppc, np, steps, ranks, omp_get_max_threads(), calls / (PetscReal)steps));

  const PetscReal s = 1e3 / steps;  // ms per step
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "prepare %9.3f  interpolate %9.3f  cn_push %9.3f  decompose %9.3f  other %9.3f  form_iteration %9.3f  field solve %9.3f  update_cells %9.3f  total %9.3f  [ms/step]\n",
    max[0] * s, max[1] * s, max[2] * s, max[3] * s, max[4] * s, max[5] * s, max[6] * s, max[7] * s, max[8] * s));

  PetscCall(simulation.finalize());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config(PetscInt n, PetscInt ppc, PetscInt steps)
{
  dx = 0.5;
  geom_nx = n;
  geom_x = geom_nx * dx;

  dt = 1.5;
  geom_nt = steps;
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "eccapfim"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {
      "Geometry",
      {
        {"x", geom_x},
        {"y", geom_x},
        {"z", geom_x},
        {"t", geom_t},
        {"dx", dx},
        {"dy", dx},
        {"dz", dx},
        {"dt", dt},
        {"diagnose_period", geom_nt},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", ppc},
        {"n", +1.0},
        {"q", -1.0},
        {"m", +1.0},
        {"T", +0.1},
      }},
    },
    {
      "Presets",
      {{
        {"command", "SetParticles"},
        {"particles", "electrons"},
        {"coordinate", {{"name", "CoordinateInBox"}}},
        {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
      }},
    },
  });
}

PetscErrorCode event_time(const char* name, PetscInt* count, PetscReal* time)
{
  PetscFunctionBeginUser;
  PetscLogEvent event;
  PetscCall(PetscLogEventGetId(name, &event));

  *count = 0;
  *time = 0;
  for (const char* stage_name :
    {"Init iteration", "Calc iteration", "After iteration"}) {
    PetscLogStage stage;
    PetscEventPerfInfo info;
    PetscCall(PetscLogStageGetId(stage_name, &stage));
    PetscCall(PetscLogEventGetPerfInfo(stage, event, &info));
    *count += info.count;
    *time += info.time;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
