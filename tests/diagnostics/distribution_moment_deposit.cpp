#include "src/diagnostics/distribution_moment.h"

#include "src/impls/basic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing \'DistributionMoment\' against the solver deposit. A lattice of  \n"
  "particles, one per cell and uniform in z, is shifted off the grid nodes. \n"
  "Moments must be the sort's Form factor deposit (periodic wrap included), \n"
  "independent of the region and the rank decomposition, also for nz = 1.   \n";

// the local width must not be smaller than the stencil (4)
constexpr PetscInt Nxy = 8;

// Momentum depends on (i, j) only, so that the plasma stays uniform in z; |p| < 1 for the histogram
Vector3R momentum(PetscInt i, PetscInt j)
{
  return Vector3R{0.05 * (i + 1), 0.04 * (j + 1), 0.03 * (i + j + 1)};
}

void overwrite_config(PetscInt nz, PetscReal shift);

PetscErrorCode diagnose(const interfaces::Particles& particles,
  const std::string& moment_name, const Region& region,
  const std::filesystem::path& dir, std::vector<float>& written);

PetscErrorCode check(PetscInt nz, PetscReal shift);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscInt procs_z = 1;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-da_processors_z", &procs_z, nullptr));

  // The shift moves the lattice off the grid nodes: 0.5 splits the weight equally
  // between two cells, 0.3 does not, which makes the cell conventions visible.
  for (PetscReal shift : {0.5, 0.3}) {
    PetscCall(check(9, shift));

    // a single cell cannot be split along z
    if (procs_z == 1) {
      PetscCall(check(1, shift));
    }
  }

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode diagnose(const interfaces::Particles& particles,
  const std::string& moment_name, const Region& region,
  const std::filesystem::path& dir, std::vector<float>& written)
{
  PetscFunctionBeginUser;
  auto&& diag = DistributionMoment::create(
    dir, particles, moment_from_string(moment_name), region);

  if (diag) {
    PetscCall(diag->diagnose(0));
    PetscCall(diag->finalize());
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  PetscInt n = region.dof;
  for (PetscInt c = 0; c < 3; ++c)
    n *= region.size[c];
  PetscCall(read_floats(dir / "0000", n, written));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check(PetscInt nz, PetscReal shift)
{
  PetscFunctionBeginUser;
  overwrite_config(nz, shift);

  basic::Simulation simulation;
  PetscCall(simulation.initialize());
  const auto& particles = *simulation.particles_[0];
  const PetscReal mpw = particles.parameters.mpw;
  const PetscInt cells = Nxy * Nxy * nz;

  auto full = [&](PetscInt dof)
  {
    return Region{
      .dim = dof > 1 ? 4 : 3,
      .dof = dof,
      .start = Vector4I{0, 0, 0, 0},
      .size = Vector4I{Nxy, Nxy, nz, dof},
    };
  };

  auto dir = [&](const char* name)
  {
    return get_out_dir(__FILE__) / std::to_string(nz) / name;
  };

  // Density: uniform, one particle per cell, total weight is `N * mpw`
  std::vector<float> density;
  PetscCall(diagnose(particles, "density", full(1), dir("density"), density));

  PetscReal total = 0.0;
  for (std::size_t i = 0; i < density.size(); i++) {
    total += density[i];
    PetscCheck(equal_rel(density[i], mpw, 1e-5, 0.0), PETSC_COMM_WORLD, PETSC_ERR_USER,
      "nz = %d, shift = %g: density is not uniform at the index %zu: %g vs %g", nz, shift, i, density[i], mpw);
  }
  PetscCheck(equal_rel(total, cells * mpw, 1e-5, 0.0), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "nz = %d, shift = %g: total weight %g differs from N * mpw = %g", nz, shift, total, cells * mpw);

  // Momentum flux against the brute-force deposit with the sort's Form factor
  constexpr PetscInt dof = 6;
  std::vector<float> flux;
  PetscCall(diagnose(particles, "momentum_flux", full(dof), dir("flux"), flux));

  const PetscInt R = (PetscInt)std::ceil(shape_radius) + 1;
  const PetscInt n[3]{Nxy, Nxy, nz};
  std::vector<PetscReal> expected(flux.size(), 0.0);

  for (PetscInt k = 0; k < nz; ++k) {
    for (PetscInt j = 0; j < Nxy; ++j) {
      for (PetscInt i = 0; i < Nxy; ++i) {
        const auto moments =
          momentum_flux(particles.parameters.m, momentum(i, j));
        for (PetscInt dk = -R; dk <= R; ++dk) {
          for (PetscInt dj = -R; dj <= R; ++dj) {
            for (PetscInt di = -R; di <= R; ++di) {
              const PetscReal w = mpw *       //
                shape_function(shift - di) *  //
                shape_function(shift - dj) *  //
                shape_function(shift - dk);
              const PetscInt x = ((i + di) % n[X] + n[X]) % n[X];
              const PetscInt y = ((j + dj) % n[Y] + n[Y]) % n[Y];
              const PetscInt z = ((k + dk) % n[Z] + n[Z]) % n[Z];
              for (PetscInt c = 0; c < dof; ++c)
                expected[indexing::t_g(x, y, z, c, dof)] += w * moments[c];
            }
          }
        }
      }
    }
  }
  for (std::size_t i = 0; i < flux.size(); i++) {
    PetscCheck(equal_rel(flux[i], expected[i], 1e-5, 1e-7), PETSC_COMM_WORLD, PETSC_ERR_USER,
      "nz = %d, shift = %g: momentum flux differs from the solver deposit at the index %zu: %g vs %g", nz, shift, i, flux[i], expected[i]);
  }

  // The slice at nz/2 equals the z-average of the full region
  Region slice = full(dof);
  slice.start[Z] = nz / 2;
  slice.size[Z] = 1;

  std::vector<float> sliced;
  PetscCall(diagnose(particles, "momentum_flux", slice, dir("slice"), sliced));

  for (PetscInt j = 0; j < Nxy; ++j) {
    for (PetscInt i = 0; i < Nxy; ++i) {
      for (PetscInt c = 0; c < dof; ++c) {
        PetscReal avg = 0.0;
        for (PetscInt k = 0; k < nz; ++k)
          avg += flux[indexing::t_g(i, j, k, c, dof)] / nz;

        PetscReal s = sliced[indexing::t_g(i, j, 0, c, dof) % (Nxy * Nxy * dof)];
        PetscCheck(equal_rel(s, avg, 1e-5, 1e-7), PETSC_COMM_WORLD, PETSC_ERR_USER,
          "nz = %d, shift = %g: slice differs from the z-average at (%d, %d, %d): %g vs %g", nz, shift, i, j, c, s, avg);
      }
    }
  }

  PetscCall(simulation.finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config(PetscInt nz, PetscReal shift)
{
  dx = dy = dz = 1.0;
  geom_nx = geom_ny = Nxy;
  geom_nz = nz;
  geom_x = geom_y = Nxy;
  geom_z = nz;

  dt = 1.0;
  geom_nt = 1;
  geom_t = 1.0;

  const char* sort_name = "particles";

  Configuration::array_t coordinates, momenta;
  for (PetscInt k = 0; k < nz; ++k) {
    for (PetscInt j = 0; j < Nxy; ++j) {
      for (PetscInt i = 0; i < Nxy; ++i) {
        const Vector3R p = momentum(i, j);
        coordinates.push_back({i + shift, j + shift, k + shift});
        momenta.push_back({p[X], p[Y], p[Z]});
      }
    }
  }

  const Configuration::array_t presets{{
    {"command", "SetParticles"},
    {"particles", sort_name},
    {"coordinate", {{"name", "PreciseCoordinate"}, {"value", coordinates}}},
    {"momentum", {{"name", "PreciseMomentum"}, {"value", momenta}}},
  }};

  Configuration::overwrite({
    {"Simulation", "basic"},
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
    {
      "Particles",
      {{
        {"sort_name", sort_name},
        {"Np", 1},
        {"n", 1.0},
        {"q", -1.0},
        {"m", 2.0},
      }},
    },
    {"Presets", presets},
  });
}
