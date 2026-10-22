#include "src/interfaces/simulation.h"
#include "src/commands/inject_particles.h"
#include "src/diagnostics/energy.h"
#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing the particle loaders: SetParticles (single and paired sorts) and\n"
  "InjectParticles for \"basic\" and \"drift_kinetic\" (both coord_is_gc). \n";

static constexpr PetscReal B_value = 2.0;

static constexpr PetscInt Np = 4;
static constexpr PetscInt inject_num = 3;

static const Vector3R r0(4.5, 4.5, 4.5);

// Signed parallel momentum is negative for the first sort
static const Vector3R p_a(0.3, 0.4, -0.5);
static const Vector3R p_b(0.1, 0.0, 0.2);

PetscErrorCode run(const std::string& simulation_name, bool coord_is_gc);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscCall(run("basic", false));
  PetscCall(run("drift_kinetic", false));
  PetscCall(run("drift_kinetic", true));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config(const std::string& simulation_name, bool coord_is_gc)
{
  dx = dy = dz = 1.0;
  geom_nx = geom_ny = geom_nz = 8;
  geom_x = geom_y = geom_z = 8.0;

  dt = 0.1;
  geom_nt = 1;
  geom_t = dt;

  const Configuration::json_t coordinate{
    {"name", "PreciseCoordinate"},
    {"value", {r0[X], r0[Y], r0[Z]}},
  };

  auto momentum = [](const Vector3R& p)
  {
    return Configuration::json_t{
      {"name", "PreciseMomentum"},
      {"value", {p[X], p[Y], p[Z]}},
    };
  };

  Configuration::overwrite({
    {"Simulation", simulation_name},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {
      "Geometry",
      {
        {"x", geom_x},
        {"y", geom_y},
        {"z", geom_z},
        {"t", geom_t},
        {"dx", dx},
        {"dy", dy},
        {"dz", dz},
        {"dt", dt},
        {"diagnose_period", dt},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {"Fields", {{"evolve", false}}},
    {"DriftKineticPush", {{"coord_is_gc", coord_is_gc}}},
    {
      "Particles",
      {
        {
          {"sort_name", "a"},
          {"Np", Np},
          {"n", +1.0},
          {"q", -1.0},
          {"m", +1.0},
          {"T", +0.1},
        },
        {
          {"sort_name", "b"},
          {"Np", Np},
          {"n", +1.0},
          {"q", +1.0},
          {"m", +4.0},
          {"T", +0.1},
        },
      },
    },
    {
      "Presets",
      {
        {
          {"command", "SetField"},
          {"field", "B"},
          {"setter",
            {
              {"name", "SetUniformField"},
              {"value", {0.0, 0.0, B_value}},
            }},
        },
        {
          {"command", "SetParticles"},
          {"particles", "a"},
          {"coordinate", coordinate},
          {"momentum", momentum(p_a)},
        },
        {
          {"command", "SetParticles"},
          {"particles", "a"},
          {"paired_with", "b"},
          {"coordinate", coordinate},
          {"momentum", momentum(p_a)},
          {"momentum_paired", momentum(p_b)},
        },
        {
          {"command", "InjectParticles"},
          {"ionized", "a"},
          {"ejected", "b"},
          {"coordinate", coordinate},
          {"per_step_particles_num", inject_num},
          {"momentum_i", momentum(p_a)},
          {"momentum_e", momentum(p_b)},
        },
        {
          {"command", "InjectParticles"},
          {"ionized", "a"},
          {"ejected", "b"},
          {"coordinate", coordinate},
          {"per_step_particles_num", inject_num},
          {"momentum_i", momentum(p_a)},
          {"momentum_e", momentum(p_b)},
        },
      },
    },
  });
}

template<typename PointT>
static PetscInt count_points(const std::vector<std::list<PointT>>& storage)
{
  PetscInt count = 0;
  for (const auto& cell : storage)
    count += (PetscInt)cell.size();
  return count;
}

PetscErrorCode check_counts(interfaces::Particles& particles, PetscInt expected)
{
  PetscFunctionBeginUser;
  // DK points are converted out of `storage`, others stay there
  auto* dk = dynamic_cast<drift_kinetic::Particles*>(&particles);

  PetscInt count = count_points(dk ? dk->get_dk_curr_storage() :  //
      std::vector<std::list<PointByField>>{});

  PetscInt staged = count_points(particles.storage);

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &staged, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

  const char* name = particles.parameters.sort_name.c_str();

  PetscCheck((dk ? count : staged) == expected, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "\"%s\": %d particles, expected %d", name, dk ? count : staged, expected);

  PetscCheck(!dk || staged == 0, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "\"%s\": %d points are left in `storage`", name, staged);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Checks every stored guiding center of `particles` against the
/// analytic conversion of the momentum `p` in `B = B_value`.
PetscErrorCode check_conversion(  //
  const drift_kinetic::Particles& particles, const Vector3R& p, bool coord_is_gc)
{
  PetscFunctionBeginUser;
  const char* name = particles.parameters.sort_name.c_str();
  const PetscReal m = particles.parameters.m;
  const PetscReal qm = particles.parameters.q / m;

  // r_gc = r + p x b / (qm B), b = z
  Vector3R expected_r = r0;
  if (!coord_is_gc)
    expected_r += Vector3R(p[Y], -p[X], 0) / (qm * B_value);

  const PetscReal expected_mu = m * (p[X] * p[X] + p[Y] * p[Y]) / (2.0 * B_value);

  for (const auto& cell : particles.get_dk_curr_storage()) {
    for (const auto& point : cell) {
      PetscCheck(equal_tol(point.r, expected_r, 1e-12), PETSC_COMM_WORLD, PETSC_ERR_USER,
        "\"%s\": wrong guiding center", name);

      PetscCheck(equal_tol(point.p_parallel, p[Z], 1e-12), PETSC_COMM_WORLD, PETSC_ERR_USER,
        "\"%s\": wrong p_parallel %g, expected %g", name, point.p_parallel, p[Z]);

      PetscCheck(equal_tol(point.mu_p, expected_mu, 1e-12), PETSC_COMM_WORLD, PETSC_ERR_USER,
        "\"%s\": wrong mu %g, expected %g", name, point.mu_p, expected_mu);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_energy(  //
  const interfaces::Particles& particles, const Vector3R& p, PetscReal energy)
{
  PetscFunctionBeginUser;
  const char* name = particles.parameters.sort_name.c_str();
  const PetscReal m = particles.parameters.m;
  const PetscReal mpw = particles.parameters.mpw;
  const PetscReal expected = inject_num * Energy::get_kinetic(p, m, mpw);

  PetscCheck(equal_tol(energy, expected, 1e-12), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "\"%s\": energy %g, expected %g", name, energy, expected);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode run(const std::string& simulation_name, bool coord_is_gc)
{
  PetscFunctionBeginUser;
  overwrite_config(simulation_name, coord_is_gc);

  std::unique_ptr<interfaces::Simulation> simulation = build_simulation();

  // The presets are executed on initialization:
  // 1 + 1 + 2 loads for "a" and 1 + 2 loads for "b"
  PetscCall(simulation->initialize());

  auto& a = simulation->get_named_particles("a");
  auto& b = simulation->get_named_particles("b");

  PetscCall(check_counts(a, Np + Np + 2 * inject_num));
  PetscCall(check_counts(b, Np + 2 * inject_num));

  if (dynamic_cast<drift_kinetic::Simulation*>(simulation.get())) {
    auto* dk_a = dynamic_cast<drift_kinetic::Particles*>(&a);
    auto* dk_b = dynamic_cast<drift_kinetic::Particles*>(&b);
    PetscCall(check_conversion(*dk_a, p_a, coord_is_gc));
    PetscCall(check_conversion(*dk_b, p_b, coord_is_gc));
  }

  // One more injection: counts add, the reported energy is the sum over the sort
  InjectParticles inject(
    {
      {a, PreciseMomentum(p_a)},
      {b, PreciseMomentum(p_b)},
    },
    0, 1, inject_num, PreciseCoordinate(r0));
  PetscCall(inject.execute(0));

  PetscCall(check_counts(a, Np + Np + 3 * inject_num));
  PetscCall(check_counts(b, Np + 3 * inject_num));
  PetscCall(check_energy(a, p_a, inject.get_ionized_energy()));
  PetscCall(check_energy(b, p_b, inject.get_ejected_energy()));

  // Outside of the window nothing is injected
  PetscCall(inject.execute(2));
  PetscCall(check_counts(a, Np + Np + 3 * inject_num));
  PetscCheck(inject.get_ionized_energy() == 0, PETSC_COMM_WORLD, PETSC_ERR_USER, "energy must be reset");

  PetscCall(simulation->finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
