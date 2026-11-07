#include "src/commands/set_particles.h"
#include "src/commands/set_particles_iaw.h"
#include "src/diagnostics/field_view.h"
#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/random_generator.h"
#include "src/utils/shape.h"
#include "tests/hybrid/hybrid_ex1_comparison.h"

static constexpr char help[] =
  "Halton/quiet versus random loading of a uniform drift-kinetic plasma.\n"
  "Ten self-consistent steps with dt=10; full 3-D density and E snapshots.\n"
  "Optional -halton_ppc 8,32,128 selects up to three even marker counts.\n";

namespace {
constexpr std::array<unsigned int, 3> seeds{1729, 2718, 31415};

// Quadratic number-density deposition on the actual DK guiding centers,
// using the same shape and periodic assembly as DK charge conservation.
PetscErrorCode collect_density(const interfaces::Particles& particles, Vec density)
{
  PetscFunctionBeginUser;
  const auto& dk = dynamic_cast<const drift_kinetic::Particles&>(particles);
  Vec local;
  PetscCall(DMGetLocalVector(particles.da_rho, &local));
  PetscCall(VecSet(local, 0));
  PetscCall(VecSet(density, 0));
  PetscReal*** arr;
  PetscCall(DMDAVecGetArrayWrite(particles.da_rho, local, &arr));
  for (const auto& cell : dk.get_dk_curr_storage()) {
    for (const auto& point : cell) {
      const Vector3R r = Shape::make_r(point.r);
      const Vector3I start{static_cast<PetscInt>(std::ceil(r[X] - 1.5)),
        static_cast<PetscInt>(std::ceil(r[Y] - 1.5)),
        static_cast<PetscInt>(std::ceil(r[Z] - 1.5))};
      for (PetscInt z = start[Z]; z < start[Z] + 3; ++z)
        for (PetscInt y = start[Y]; y < start[Y] + 3; ++y)
          for (PetscInt x = start[X]; x < start[X] + 3; ++x)
            arr[z][y][x] += particles.parameters.mpw *
              spline_of_2nd_order(r[X] - x) * spline_of_2nd_order(r[Y] - y) *
              spline_of_2nd_order(r[Z] - z);
    }
  }
  PetscCall(DMDAVecRestoreArrayWrite(particles.da_rho, local, &arr));
  PetscCall(DMLocalToGlobal(particles.da_rho, local, ADD_VALUES, density));
  PetscCall(DMRestoreLocalVector(particles.da_rho, &local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Save number density with the same quadratic shape used for charge in the
// field solver, without the lower-order DistributionMoment diagnostic.
class DensityView : public interfaces::Diagnostic {
public:
  DensityView(const interfaces::Particles& particles, const std::string& path)
    : particles(particles)
  {
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(particles.rho[1], &density));
    const Region region{3, 1, {0, 0, 0, 0}, {geom_nx, geom_ny, geom_nz, 1}};
    view = FieldView::create(path, particles.da_rho, density, region);
  }

  PetscErrorCode diagnose(PetscInt t) override
  {
    PetscFunctionBeginUser;
    PetscCall(collect_density(particles, density));
    PetscCall(view->diagnose(t));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode finalize() override
  {
    PetscFunctionBeginUser;
    PetscCall(view->finalize());
    PetscCall(VecDestroy(&density));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  const interfaces::Particles& particles;
  Vec density;
  std::unique_ptr<FieldView> view;
};

// Customize only initialization to pass an exact PAIR count to the existing
// IAW loader. Motion, field solve, charge deposition and time loop are DK.
class NoiseSimulation : public drift_kinetic::Simulation {
public:
  explicit NoiseSimulation(bool halton) : halton(halton) {}
  PetscReal electric_square_sum = 0;
  PetscReal initial_density_rms = 0;

  PetscErrorCode check_initial_density()
  {
    PetscFunctionBeginUser;
    auto& electrons = get_named_particles("electrons");
    Vec error;
    PetscCall(VecDuplicate(electrons.rho[1], &error));
    PetscCall(collect_density(electrons, error));
    PetscScalar sum;
    PetscCall(VecSum(error, &sum));
    const PetscReal cells = geom_nx * geom_ny * geom_nz;
    PetscCheck(std::abs(PetscRealPart(sum) / cells - 1.0) < 1e-12,
      PETSC_COMM_WORLD, PETSC_ERR_USER, "Mean number density must equal 1");
    PetscCall(VecShift(error, -1.0));
    PetscCall(VecNorm(error, NORM_2, &initial_density_rms));
    initial_density_rms /= std::sqrt(cells);
    PetscCall(VecDestroy(&error));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

protected:
  PetscErrorCode initialize_implementation() override
  {
    PetscFunctionBeginUser;
    PetscCall(drift_kinetic::Simulation::initialize_implementation());
    PetscCall(VecSet(E, 0));
    PetscCall(VecSet(B, 0));
    PetscCall(VecStrideSet(B, Z, ex1::B_value));
    PetscCall(VecCopy(B, B0));
    auto& ions = get_named_particles("ions");
    auto& electrons = get_named_particles("electrons");
    const PetscInt count = geom_nx * geom_ny * geom_nz * ions.parameters.Np;
    if (halton) {
      // Both species share guiding centers; E=0 initially satisfies Gauss's
      // law. The unperturbed IAW loader supplies Halton/stratified positions
      // and antithetic, low-discrepancy Maxwellian velocities.
      for (auto* sort : {&ions, &electrons}) {
        SetParticlesIonAcousticWave loader(*sort, count / 2,
          0.0, 0.01, 0.01, {0, 0, 1}, {}, {}, {});
        PetscCall(loader.execute(0));
      }
    }
    else {
      SetParticles loader({
          {ions, MaxwellianMomentum{ions.parameters, true}},
          {electrons, MaxwellianMomentum{electrons.parameters, true}}}, count,
        CoordinateInBox{{{0, 0, 0}, {geom_x, geom_y, geom_z}}});
      PetscCall(loader.execute(0));
    }
    for (const auto* sort : {&ions, &electrons}) {
      const auto& dk = dynamic_cast<const drift_kinetic::Particles&>(*sort);
      PetscInt actual = 0;
      for (const auto& cell : dk.get_dk_curr_storage())
        actual += static_cast<PetscInt>(cell.size());
      PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &actual, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
      PetscCheck(actual == count, PETSC_COMM_WORLD, PETSC_ERR_USER, "Unequal marker counts");
      diagnostics_.push_back(std::make_unique<DensityView>(*sort,
        CONFIG().out_dir + "/" + sort->parameters.sort_name + "/density"));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode timestep_implementation(PetscInt t) override
  {
    PetscFunctionBeginUser;
    PetscCall(drift_kinetic::Simulation::timestep_implementation(t));
    PetscReal norm;
    PetscCall(VecNorm(E, NORM_2, &norm));
    PetscCheck(std::isfinite(norm), PETSC_COMM_WORLD, PETSC_ERR_USER, "Non-finite electric field");
    electric_square_sum += norm * norm;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  bool halton;
};

PetscErrorCode run_case(PetscInt ppc, bool halton, unsigned int seed,
  PetscReal& density_rms, PetscReal& electric_rms)
{
  PetscFunctionBeginUser;
  // Same physical plasma as hybrid_ex1: periodic 3^3 cells, dx=10,
  // Bz=1, Te=Ti=0.1 keV, mi/me=100, n_e=n_i=1.
  PetscCall(ex1::overwrite_config(__FILE__, "drift_kinetic", "drift_kinetic", "drift_kinetic", 10.0));
  auto config = CONFIG().json;
  config["Geometry"]["dt"] = 10.0;
  config["Geometry"]["t"] = 100.0;
  config["Geometry"]["diagnose_period"] = 10.0;
  for (auto& sort : config["Particles"])
    sort["Np"] = ppc;
  config["Presets"] = Configuration::json_t::array();
  const auto name = halton ? std::format("halton/ppc_{}", ppc)
    : std::format("random/ppc_{}/seed_{}", ppc, seed);
  const auto path = get_out_dir(__FILE__) / name;
  config["OutputDirectory"] = path;
  config["Sampling"] = {{"algorithm", halton ? "halton_quiet" : "random"},
    {"seed", seed}, {"markers_per_cell", ppc}};
  config["Diagnostics"] = {{{"diagnostic", "FieldView"}, {"field", "E"}}};
  config["DriftKineticPush"]["fail_on_final_nonconvergence"] = true;
  Configuration::overwrite(std::move(config));
  Configuration::save(path);
  RandomGenerator::get().seed(seed);
  NoiseSimulation simulation(halton);
  PetscCall(simulation.initialize());
  PetscCall(simulation.check_initial_density());
  density_rms = simulation.initial_density_rms;
  PetscCall(simulation.calculate());
  // RMS of the vector magnitude over all grid nodes and the TEN evolved
  // frames (exclude the exactly zero initial field).
  electric_rms = std::sqrt(simulation.electric_square_sum /
    (geom_nt * geom_nx * geom_ny * geom_nz));
  PetscCall(simulation.finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode run()
{
  PetscFunctionBeginUser;
  std::array<PetscInt, 3> particle_counts{8, 16, 32};
  PetscInt number_of_counts = particle_counts.size();
  PetscBool specified_counts;
  PetscCall(PetscOptionsGetIntArray(nullptr, nullptr, "-halton_ppc",
    particle_counts.data(), &number_of_counts, &specified_counts));
  if (!specified_counts)
    number_of_counts = particle_counts.size();
  PetscCheck(number_of_counts > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "-halton_ppc requires one to three positive even counts");
  // Native DK tolerance is far below the measured noise; tighter ex1 energy
  // tolerances add many iterations without improving this comparison.
  for (const char* option : {"-snes_atol", "-snes_rtol"}) {
    PetscBool specified;
    PetscCall(PetscOptionsHasName(nullptr, nullptr, option, &specified));
    if (!specified)
      PetscCall(PetscOptionsSetValue(nullptr, option, "1e-10"));
  }
  PetscCall(PetscOptionsSetValue(nullptr, "-snes_stol", "0"));
  Configuration::json_t completed_cases = Configuration::json_t::array();
  for (PetscInt i = 0; i < number_of_counts; ++i) {
    const auto ppc = particle_counts[i];
    PetscCheck(ppc > 0 && ppc % 2 == 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "Marker counts must be positive and even, got %d", ppc);
    for (PetscInt j = 0; j < i; ++j)
      PetscCheck(ppc != particle_counts[j], PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
        "Marker counts must be distinct");
    PetscReal halton_density, halton_electric;
    PetscCall(run_case(ppc, true, 0, halton_density, halton_electric));
    completed_cases.push_back(std::format("halton/ppc_{}", ppc));
    for (const auto seed : seeds) {
      PetscReal random_density, random_electric;
      PetscCall(run_case(ppc, false, seed, random_density, random_electric));
      PetscCheck(halton_density < random_density, PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Initial Halton density noise must be below random (ppc=%d seed=%u)", ppc, seed);
      PetscCheck(halton_electric < random_electric, PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Halton E RMS %g must be below random %g (ppc=%d seed=%u)",
        halton_electric, random_electric, ppc, seed);
      completed_cases.push_back(std::format("random/ppc_{}/seed_{}", ppc, seed));
    }
  }
  // Analyze exactly this successful scan, even if older scans remain on disk.
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0) {
    std::ofstream manifest(get_out_dir(__FILE__) / "cases.json");
    manifest << completed_cases.dump(2) << '\n';
    manifest.close();
    PetscCheck(manifest.good(), PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Cannot write cases.json");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
}  // namespace

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(run());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
