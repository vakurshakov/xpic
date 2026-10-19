#include "distribution_moment.h"

#include "src/utils/geometries.h"
#include "src/utils/shape.h"
#include "src/utils/utils.h"
#include "src/utils/vector_utils.h"


std::unique_ptr<DistributionMoment> DistributionMoment::create(
  const std::string& out_dir, const interfaces::Particles& particles,
  const Moment& moment, const Region& region)
{
  PetscFunctionBeginUser;
  MPI_Comm newcomm;
  PetscCallAbort(PETSC_COMM_WORLD, World::create_local_comm(particles.world.da, region, &newcomm));
  if (newcomm == MPI_COMM_NULL)
    PetscFunctionReturn(nullptr);

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
  PetscCall(World::create_local_dm(da_glob, reg, comm, &da));
  PetscCall(DMCreateGlobalVector(da, &field));
  PetscCall(DMCreateLocalVector(da, &field_loc));
  PetscCall(FieldView::set_data_views(reg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DistributionMoment::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(FieldView::finalize());
  PetscCall(VecDestroy(&field));
  PetscCall(VecDestroy(&field_loc));
  PetscCall(DMDestroy(&da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DistributionMoment::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t % diagnose_period_ == 0) {
    PetscCall(collect());
    PetscCall(FieldView::diagnose(t));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief `Shape` is reduced from global `Shape` to be used in a specific task:
 * (1) All moments are represented as a cell-centered quantities;
 * (2) Lower order `sfunc` is used to reduce the computational time;
 * (3) Storage size for `cache` is also reduced using shape products.
 */
void DistributionMoment::Shape::setup(const Vector3R& r)
{
  static constexpr const auto& sfunc = spline_of_1st_order;

  Vector3R p_r = ::Shape::make_r(r);
  start = ::Shape::make_start(p_r, shr);

  static const double offset[3]{
    geom_nx == 1 ? 0.5 : 0,
    geom_ny == 1 ? 0.5 : 0,
    geom_nz == 1 ? 0.5 : 0,
  };

#pragma omp simd
  for (PetscInt i = 0; i < shm; ++i) {
    auto g_x = (PetscReal)(start[X] + i % shw) + offset[X];
    auto g_y = (PetscReal)(start[Y] + (i / shw) % shw) + offset[Y];
    auto g_z = (PetscReal)(start[Z] + (i / shw) / shw) + offset[Z];
    cache[i] = sfunc(p_r[X] - g_x) * sfunc(p_r[Y] - g_y) * sfunc(p_r[Z] - g_z);
  }
}

PetscErrorCode DistributionMoment::collect()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(field, 0));
  PetscCall(VecSet(field_loc, 0));

  PetscReal**** arr;
  PetscCall(DMDAVecGetArrayDOFWrite(da, field_loc, &arr));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da_glob, REP3_A(&start), REP3_A(&size)));

  const Vector3I gstart = vector_cast(region.start);
  const Vector3I gsize = vector_cast(region.size);

  Shape shape;

#pragma omp parallel for private(shape)
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    Vector3I vg{
      start[X] + g % size[X],
      start[Y] + (g / size[X]) % size[Y],
      start[Z] + (g / size[X]) / size[Y],
    };

    if (!is_point_within_bounds(vg, gstart, gsize))
      continue;

    for (auto&& point : particles.storage[g]) {
      shape.setup(point.r);

      std::vector<PetscReal> moments = moment(particles, point);
      auto msize = static_cast<PetscInt>(moments.size());

      for (PetscInt i = 0; i < shape.shm; ++i) {
        PetscInt g_x = shape.start[X] + i % shape.shw;
        PetscInt g_y = shape.start[Y] + (i / shape.shw) % shape.shw;
        PetscInt g_z = shape.start[Z] + (i / shape.shw) / shape.shw;

        PetscReal si = shape.cache[i] * particles.n_Np(point);

        for (PetscInt j = 0; j < msize; ++j) {
          PetscReal mj = moments[j] * si;

#pragma omp atomic update
          arr[g_z][g_y][g_x][j] += mj;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOFWrite(da, field_loc, &arr));
  PetscCall(DMLocalToGlobal(da, field_loc, ADD_VALUES, field));
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
