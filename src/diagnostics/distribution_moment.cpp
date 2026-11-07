#include "distribution_moment.h"

#include "src/utils/utils.h"


std::unique_ptr<DistributionMoment> DistributionMoment::create(
  const std::string& out_dir, const interfaces::Particles& particles,
  const Moment& moment, const Region& region)
{
  PetscFunctionBeginUser;
  MPI_Comm newcomm;
  PetscCallAbort(PETSC_COMM_WORLD, World::create_local_comm(particles.world.da, region, &newcomm));

  auto* diagnostic = new DistributionMoment(out_dir, particles, moment, newcomm);
  PetscCallAbort(PETSC_COMM_WORLD, diagnostic->set_data_views(region));
  PetscFunctionReturn(std::unique_ptr<DistributionMoment>(diagnostic));
}

DistributionMoment::DistributionMoment(const interfaces::Particles& particles)
  : FieldView(particles.world.da, nullptr), particles(particles)
{
}

DistributionMoment::DistributionMoment(const std::string& out_dir,
  const interfaces::Particles& particles, const Moment& moment, MPI_Comm newcomm)
  : FieldView(out_dir, particles.world.da, nullptr, newcomm),
    particles(particles),
    moment(moment)
{
}

PetscErrorCode DistributionMoment::set_data_views(const Region& reg)
{
  PetscFunctionBeginUser;
  da_glob = da;
  region = reg;

  const Region world_region{
    .dim = reg.dim,
    .dof = reg.dof,
    .start = Vector4I{0, 0, 0, 0},
    .size = Vector4I{geom_nx, geom_ny, geom_nz, reg.dof},
  };
  PetscCall(World::create_local_dm(da_glob, world_region, PETSC_COMM_WORLD, &da_scratch));
  PetscCall(DMCreateGlobalVector(da_scratch, &scratch));
  PetscCall(DMCreateLocalVector(da_scratch, &scratch_loc));

  // Only the ranks touching the region own the output
  da = nullptr;
  if (comm != MPI_COMM_NULL) {
    PetscCall(World::create_local_dm(da_glob, reg, comm, &da));
    PetscCall(DMCreateGlobalVector(da, &field));
    PetscCall(FieldView::set_data_views(reg));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DistributionMoment::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(FieldView::finalize());
  PetscCall(VecDestroy(&field));
  PetscCall(VecDestroy(&scratch));
  PetscCall(VecDestroy(&scratch_loc));
  PetscCall(DMDestroy(&da));
  PetscCall(DMDestroy(&da_scratch));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DistributionMoment::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t % diagnose_period_ == 0) {
    PetscCall(collect());
    if (comm != MPI_COMM_NULL)
      PetscCall(FieldView::diagnose(t));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode DistributionMoment::collect()
{
  PetscFunctionBeginUser;
  ParticleStorage::Attributes f;
  PetscCall(particles.storage.get_attributes(f));
  const PointArrays arrays(f);
  const std::vector<PetscInt>& offsets = particles.storage.cell_offsets();

  PetscCall(collect([&](PetscInt g, auto&& deposit)
  {
    for (PetscInt i = offsets[g]; i < offsets[g + 1]; i++) {
      const Point point = arrays.load(i);
      deposit(point.r, moment(particles, point));
    }
  }));

  PetscCall(particles.storage.restore_attributes(f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

Vector3R to_cylindrical(const Vector3R& r, const Vector3R& v)
{
  PetscReal x = r[X] - 0.5 * geom_x;
  PetscReal y = r[Y] - 0.5 * geom_y;
  PetscReal rho = std::hypot(x, y);

  // Particles close to r=0 are not taken into account
  if (std::isinf(1.0 / rho))
    return v;

  return {
    (+x * v[X] + y * v[Y]) / rho,
    (-y * v[X] + x * v[Y]) / rho,
    v[Z],
  };
}

inline std::vector<PetscReal> get_density(
  const interfaces::Particles& /* particles */, const Point& /* point */)
{
  return {1.0};
}

inline std::vector<PetscReal> get_current(
  const interfaces::Particles& particles, const Point& point)
{
  auto&& q = particles.parameters.q;
  auto&& v = point.p;
  return {q * v[X], q * v[Y], q * v[Z]};
}

std::vector<PetscReal> momentum_flux(PetscReal m, const Vector3R& v)
{
  return {
    m * v[X] * v[X],
    m * v[X] * v[Y],
    m * v[X] * v[Z],
    m * v[Y] * v[Y],
    m * v[Y] * v[Z],
    m * v[Z] * v[Z],
  };
}

std::vector<PetscReal> momentum_flux_diag(PetscReal m, const Vector3R& v)
{
  return {
    m * v[X] * v[X],
    m * v[Y] * v[Y],
    m * v[Z] * v[Z],
  };
}

std::vector<PetscReal> momentum_flux_cyl(
  PetscReal m, const Vector3R& r, const Vector3R& v)
{
  return momentum_flux(m, to_cylindrical(r, v));
}

std::vector<PetscReal> momentum_flux_diag_cyl(
  PetscReal m, const Vector3R& r, const Vector3R& v)
{
  return momentum_flux_diag(m, to_cylindrical(r, v));
}

inline std::vector<PetscReal> get_momentum_flux(
  const interfaces::Particles& particles, const Point& point)
{
  return momentum_flux(particles.parameters.m, point.p);
}

inline std::vector<PetscReal> get_momentum_flux_diag(
  const interfaces::Particles& particles, const Point& point)
{
  return momentum_flux_diag(particles.parameters.m, point.p);
}

inline std::vector<PetscReal> get_momentum_flux_cyl(
  const interfaces::Particles& particles, const Point& point)
{
  return momentum_flux_cyl(particles.parameters.m, point.r, point.p);
}

inline std::vector<PetscReal> get_momentum_flux_diag_cyl(
  const interfaces::Particles& particles, const Point& point)
{
  return momentum_flux_diag_cyl(particles.parameters.m, point.r, point.p);
}


Moment moment_from_string(const std::string& name)
{
  if (name == "density")
    return get_density;
  if (name == "current")
    return get_current;
  if (name == "momentum_flux")
    return get_momentum_flux;
  if (name == "momentum_flux_cyl")
    return get_momentum_flux_cyl;
  if (name == "momentum_flux_diag")
    return get_momentum_flux_diag;
  if (name == "momentum_flux_diag_cyl")
    return get_momentum_flux_diag_cyl;

  throw std::runtime_error("Unkown moment name " + std::string(name));
}
