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
  PetscCallAbort(PETSC_COMM_WORLD, get_local_communicator(particles.world.da, region, &newcomm));
  if (newcomm == MPI_COMM_NULL)
    PetscFunctionReturn(nullptr);

  auto* diagnostic = new DistributionMoment(out_dir, particles, moment, newcomm);
  PetscCallAbort(PETSC_COMM_WORLD, diagnostic->set_data_views(region));
  PetscFunctionReturn(std::unique_ptr<DistributionMoment>(diagnostic));
}

DistributionMoment::DistributionMoment(const interfaces::Particles& particles)
  : FieldView(particles.world.da, nullptr), particles_(particles)
{
}

DistributionMoment::DistributionMoment(const std::string& out_dir,
  const interfaces::Particles& particles, const Moment& moment, MPI_Comm newcomm)
  : FieldView(out_dir, particles.world.da, nullptr, newcomm),
    particles_(particles),
    moment_(moment)
{
}

PetscErrorCode DistributionMoment::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(FieldView::finalize());
  PetscCall(VecDestroy(&local_));
  PetscCall(VecDestroy(&field_));
  PetscCall(DMDestroy(&da_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DistributionMoment::set_data_views(const Region& region)
{
  PetscFunctionBeginUser;
  PetscCall(set_local_da(region));
  PetscCall(DMCreateLocalVector(da_, &local_));
  PetscCall(DMCreateGlobalVector(da_, &field_));

  PetscCall(FieldView::set_data_views(region));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @note First and foremost, for `ChargeConservation` diagnostic this can be avoided!
PetscErrorCode DistributionMoment::set_local_da(const Region& region)
{
  PetscFunctionBeginUser;
  global_da_ = da_;

  Vector3I g_start = vector_cast(region.start);
  Vector3I g_size = vector_cast(region.size);
  Vector3I g_end = g_start + g_size;

  PetscInt s;
  DMDAStencilType st;
  PetscInt size[3];
  PetscInt proc[3];
  DMBoundaryType bound[3];
  PetscCall(DMDAGetInfo(global_da_, nullptr, REP3_A(&size), REP3_A(&proc), nullptr, &s, REP3_A(&bound), &st));

  const PetscInt* ownership[3];
  PetscCall(DMDAGetOwnershipRanges(global_da_, REP3_A(&ownership)));

  PetscInt l_proc[3];
  DMBoundaryType l_bound[3];
  std::vector<PetscInt> l_ownership[3];

  // Collecting number of processes and ownership ranges using global DMDA
  for (PetscInt i = 0; i < 3; ++i) {
    l_proc[i] = 0;

    PetscInt start = 0, end = 0;

    for (PetscInt s = 0; s < proc[i]; ++s) {
      end += ownership[i][s];

      if (g_start[i] < end && start < g_end[i]) {
        l_proc[i]++;

        PetscInt l_si = std::max(g_start[i], start);
        PetscInt l_ei = std::min(g_end[i], end);

        l_ownership[i].emplace_back(l_ei - l_si);
      }

      start += ownership[i][s];
    }

    // Mimic global boundaries, if we touch them
    l_bound[i] = (g_size[i] == size[i]) ? bound[i] : DM_BOUNDARY_GHOSTED;
  }

  PetscCall(DMDACreate3d(comm_, REP3_A(l_bound), st, REP3_A(g_size), REP3_A(l_proc), region.dof, s, l_ownership[X].data(), l_ownership[Y].data(), l_ownership[Z].data(), &da_));
  PetscCall(DMDASetOffset(da_, REP3_A(g_start), REP3(0)));
  PetscCall(DMSetUp(da_));
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
 * @brief This class is reduced from global `Shape` to be used in a specific task:
 * (1) All moments are represented as a cell-centered quantities;
 * (2) Lower order `sfunc` is used to reduce the computational time;
 * (3) Storage size for `cache` is also reduced using shape products.
 */
struct DistributionMoment::Shape {
  Shape() = default;

  static constexpr PetscInt shr = 1;
  static constexpr PetscInt shw = 2;
  static constexpr PetscInt shm = POW3(shw);
  static constexpr const auto& sfunc = spline_of_1st_order;

  Vector3I start;
  PetscReal cache[shm];

  void setup(const Vector3R& r)
  {
    Vector3R p_r = ::Shape::make_r(r);
    start = ::Shape::make_start(p_r, shr);

#pragma omp simd
    for (PetscInt i = 0; i < shm; ++i) {
      auto g_x = (PetscReal)(start[X] + i % shw) + 0.5;
      auto g_y = (PetscReal)(start[Y] + (i / shw) % shw) + 0.5;
      auto g_z = (PetscReal)(start[Z] + (i / shw) / shw) + 0.5;
      cache[i] = sfunc(p_r[X] - g_x) * sfunc(p_r[Y] - g_y) * sfunc(p_r[Z] - g_z);
    }
  }
};

PetscErrorCode DistributionMoment::collect()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(local_, 0.0));
  PetscCall(VecSet(field_, 0.0));

  PetscReal**** arr;
  PetscCall(DMDAVecGetArrayDOFWrite(da_, local_, &arr));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(global_da_, REP3_A(&start), REP3_A(&size)));

  const Vector3I gstart = vector_cast(region_.start);
  const Vector3I gsize = vector_cast(region_.size);

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

    for (auto&& point : particles_.storage[g]) {
      shape.setup(point.r);

      std::vector<PetscReal> moments = moment_(particles_, point);
      auto msize = static_cast<PetscInt>(moments.size());

      for (PetscInt i = 0; i < shape.shm; ++i) {
        PetscInt g_x = shape.start[X] + i % shape.shw;
        PetscInt g_y = shape.start[Y] + (i / shape.shw) % shape.shw;
        PetscInt g_z = shape.start[Z] + (i / shape.shw) / shape.shw;

        PetscReal si = shape.cache[i] * particles_.n_Np(point);

        for (PetscInt j = 0; j < msize; ++j) {
          PetscReal mj = moments[j] * si;

#pragma omp atomic update
          arr[g_z][g_y][g_x][j] += mj;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOFWrite(da_, local_, &arr));
  PetscCall(DMLocalToGlobal(da_, local_, ADD_VALUES, field_));
  PetscFunctionReturn(PETSC_SUCCESS);
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

inline std::vector<PetscReal> get_momentum_flux(
  const interfaces::Particles& particles, const Point& point)
{
  auto&& m = particles.parameters.m;
  auto&& v = point.p;
  return {
    m * v[X] * v[X],
    m * v[X] * v[Y],
    m * v[X] * v[Z],
    m * v[Y] * v[Y],
    m * v[Y] * v[Z],
    m * v[Z] * v[Z],
  };
}

inline std::vector<PetscReal> get_momentum_flux_diag(
  const interfaces::Particles& particles, const Point& point)
{
  auto&& m = particles.parameters.m;
  auto&& v = point.p;
  return {
    m * v[X] * v[X],
    m * v[Y] * v[Y],
    m * v[Z] * v[Z],
  };
}

inline std::vector<PetscReal> _get_v_cyl(const Point& point)
{
  PetscReal x = point.x() - 0.5 * geom_x;
  PetscReal y = point.y() - 0.5 * geom_y;
  PetscReal r = std::hypot(x, y);

  auto&& v = point.p;

  // Particles close to r=0 are not taken into account
  if (std::isinf(1.0 / r))
    return {v[X], v[Y], v[Z]};

  return {
    (+x * v[X] + y * v[Y]) / r,
    (-y * v[X] + x * v[Y]) / r,
    v[Z],
  };
}

inline std::vector<PetscReal> get_momentum_flux_cyl(
  const interfaces::Particles& particles, const Point& point)
{
  auto&& m = particles.parameters.m;
  auto&& v = _get_v_cyl(point);
  return {
    m * v[R] * v[R],
    m * v[R] * v[A],
    m * v[R] * v[Z],
    m * v[A] * v[A],
    m * v[A] * v[Z],
    m * v[Z] * v[Z],
  };
}

inline std::vector<PetscReal> get_momentum_flux_diag_cyl(
  const interfaces::Particles& particles, const Point& point)
{
  auto&& m = particles.parameters.m;
  auto&& v = _get_v_cyl(point);
  return {
    m * v[R] * v[R],
    m * v[A] * v[A],
    m * v[Z] * v[Z],
  };
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
