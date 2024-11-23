#include "distribution_moment.h"

#include "src/utils/shape.h"
#include "src/utils/region_operations.h"
#include "src/utils/utils.h"
#include "src/utils/vector_utils.h"

/// @note Do not move this into header file, this will pollute global namespace
using Particles = interfaces::Particles;


std::unique_ptr<Distribution_moment> Distribution_moment::create(
  const std::string& out_dir, const Particles& particles, const Moment& moment,
  const Region& region)
{
  PetscFunctionBeginUser;
  MPI_Comm newcomm;
  PetscCallThrow(get_local_communicator(particles.world_.da, region, &newcomm));
  if (newcomm == MPI_COMM_NULL)
    PetscFunctionReturn(nullptr);

  auto* diagnostic = new Distribution_moment(out_dir, particles, moment, newcomm);
  PetscCallThrow(diagnostic->set_data_views(region));
  PetscFunctionReturn(std::unique_ptr<Distribution_moment>(diagnostic));
}


Distribution_moment::Distribution_moment(const std::string& out_dir,
  const Particles& particles, const Moment& moment, MPI_Comm newcomm)
  : Field_view(out_dir, particles.world_.da, nullptr, newcomm),
    particles_(particles),
    moment_(moment)
{
}


Distribution_moment::~Distribution_moment()
{
  PetscFunctionBeginUser;
  PetscCallVoid(DMDestroy(&da_));
  PetscCallVoid(VecDestroy(&local_));
  PetscCallVoid(VecDestroy(&field_));
  PetscFunctionReturnVoid();
}


PetscErrorCode Distribution_moment::set_data_views(const Region& region)
{
  PetscFunctionBeginUser;
  PetscCall(set_da(region));
  PetscCall(DMCreateLocalVector(da_, &local_));
  PetscCall(DMCreateGlobalVector(da_, &field_));

  PetscCall(Field_view::set_data_views(region));

  // Later we'll use `Node` structure that uses shifted coordinates
  region_.start -= static_cast<PetscInt>(std::ceil(shape_radius));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Distribution_moment::set_da(const Region& region)
{
  PetscFunctionBeginUser;
  Vector3I g_start = vector_cast(region.start);
  Vector3I g_size = vector_cast(region.size);
  Vector3I g_end = g_start + g_size;

  PetscInt dim, s;
  DMDAStencilType st;
  PetscInt size[3];
  PetscInt proc[3];
  DMBoundaryType bound[3];
  PetscCall(DMDAGetInfo(da_, &dim, REP3_A(&size), REP3_A(&proc), nullptr, &s, REP3_A(&bound), &st));

  const PetscInt* ownership[3];
  PetscCall(DMDAGetOwnershipRanges(da_, REP3_A(&ownership)));

  PetscInt l_proc[3];
  DMBoundaryType l_bound[3];
  std::vector<PetscInt> l_ownership[3];

  // Collecting number of processes and ownership ranges using global DMDA
  for (PetscInt i = 0; i < dim; ++i) {
    PetscInt start = 0;
    PetscInt end = ownership[i][0];

    for (PetscInt s = 0; s < proc[i]; ++s) {
      if (start < g_end[i] && end > g_start[i]) {
        l_proc[i]++;
        l_ownership[i].emplace_back(
          std::min(g_end[i], end) - std::max(g_start[i], start));
      }
      start += ownership[i][s];
      end += start;
    }

    // Mimic global boundaries, if we touch them
    l_bound[i] = (g_size[i] == size[i]) ? bound[i] : DM_BOUNDARY_GHOSTED;
  }

  PetscCall(DMDACreate3d(comm_, REP3_A(l_bound), st, REP3_A(g_size), REP3_A(l_proc), 1, s, l_ownership[X].data(), l_ownership[Y].data(), l_ownership[Z].data(), &da_));
  PetscCall(DMDASetOffset(da_, REP3_A(g_start), 0, 0, 0));
  PetscCall(DMSetUp(da_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Distribution_moment::diagnose(timestep_t t)
{
  PetscFunctionBeginUser;
  if (t % diagnose_period == 0) {
    PetscCall(collect());
    PetscCall(Field_view::diagnose(t));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @note Communication is needed to prevent data losses. Moreover, the type of
 * data exchanges is dictated by the coordinates of the moment (`Projector`).
 * For example, if we collect particles density on (x, y, z) coordinates, we
 * need to exchange the values that are placed in ghost cells. But if we are
 * collecting velocity distribution on (Vx, Vy, Vz) coordinates, the equivalent
 * of `MPI_Allreduce()` operation is needed.
 */
PetscErrorCode Distribution_moment::collect()
{
  PetscFunctionBeginUser;

  PetscCall(VecSet(local_, 0.0));
  PetscCall(VecSet(field_, 0.0));

  PetscReal*** arr;
  PetscCall(DMDAVecGetArrayWrite(da_, local_, &arr));

#pragma omp parallel for
  for (const Point& point : particles_.points()) {
    /// @todo We can reuse `Simple_decomposition`
    /// algorithm here and make `Shape::make*()` private
    const Vector3R p_r = Shape::make_r(point.r);
    const Vector3I p_g = Shape::make_g(p_r);

    if (!is_point_within_bounds(
          p_g, vector_cast(region_.start), vector_cast(region_.size)))
      continue;

    PetscReal n = particles_.density(point) / particles_.particles_number(point);

    // clang-format off
    for (PetscInt z = 0; z < shape_width; ++z) {
    for (PetscInt y = 0; y < shape_width; ++y) {
    for (PetscInt x = 0; x < shape_width; ++x) {
      PetscInt g_x = p_g[X] + x;
      PetscInt g_y = p_g[Y] + y;
      PetscInt g_z = p_g[Z] + z;

#pragma omp atomic update
      arr[g_z][g_y][g_x] += moment_.get(particles_, point) * n *
        shape_function(p_r[X] - g_x) *
        shape_function(p_r[Y] - g_y) *
        shape_function(p_r[Z] - g_z);
    }}}
    // clang-format on
  }
  PetscCall(DMDAVecRestoreArrayWrite(da_, local_, &arr));
  PetscCall(DMLocalToGlobal(da_, local_, ADD_VALUES, field_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


inline PetscReal get_zeroth(const Particles&, const Point&)
{
  return 1.0;
}

inline PetscReal get_Vx(const Particles& particles, const Point& point)
{
  return particles.velocity(point).x();
}

inline PetscReal get_Vy(const Particles& particles, const Point& point)
{
  return particles.velocity(point).y();
}

inline PetscReal get_Vz(const Particles& particles, const Point& point)
{
  return particles.velocity(point).z();
}

inline PetscReal get_mVxVx(const Particles& particles, const Point& point)
{
  return particles.mass(point) * get_Vx(particles, point) *
    get_Vx(particles, point);
}

inline PetscReal get_mVxVy(const Particles& particles, const Point& point)
{
  return particles.mass(point) * get_Vx(particles, point) *
    get_Vy(particles, point);
}

inline PetscReal get_mVxVz(const Particles& particles, const Point& point)
{
  return particles.mass(point) * get_Vx(particles, point) *
    get_Vz(particles, point);
}

inline PetscReal get_mVyVy(const Particles& particles, const Point& point)
{
  return particles.mass(point) * get_Vy(particles, point) *
    get_Vy(particles, point);
}

inline PetscReal get_mVyVz(const Particles& particles, const Point& point)
{
  return particles.mass(point) * get_Vy(particles, point) *
    get_Vz(particles, point);
}

inline PetscReal get_mVzVz(const Particles& particles, const Point& point)
{
  return particles.mass(point) * get_Vz(particles, point) *
    get_Vz(particles, point);
}

inline PetscReal get_Vr(const Particles& particles, const Point& point)
{
  PetscReal x = point.x() - 0.5 * geom_x;
  PetscReal y = point.y() - 0.5 * geom_y;
  PetscReal r = sqrt(x * x + y * y);

  // Particles close to r=0 are not taken into account
  if (std::isinf(1.0 / r))
    return 0.0;

  return (+x * get_Vx(particles, point) + y * get_Vy(particles, point)) / r;
}

inline PetscReal get_Vphi(const Particles& particles, const Point& point)
{
  PetscReal x = point.x() - 0.5 * geom_x;
  PetscReal y = point.y() - 0.5 * geom_y;
  PetscReal r = sqrt(x * x + y * y);

  // Particles close to r=0 are not taken into account
  if (std::isinf(1.0 / r))
    return 0.0;

  return (-y * get_Vx(particles, point) + x * get_Vy(particles, point)) / r;
}

inline PetscReal get_mVrVr(const Particles& particles, const Point& point)
{
  return particles.mass(point) * get_Vr(particles, point) *
    get_Vr(particles, point);
}

inline PetscReal get_mVrVphi(const Particles& particles, const Point& point)
{
  return particles.mass(point) * get_Vr(particles, point) *
    get_Vphi(particles, point);
}

inline PetscReal get_mVphiVphi(const Particles& particles, const Point& point)
{
  return particles.mass(point) * get_Vphi(particles, point) *
    get_Vphi(particles, point);
}


Moment::Moment(const Particles& particles, const getter& get)
  : particles_(particles), get(get)
{
}

Moment Moment::from_string(const Particles& particles, const std::string& name)
{
  getter get = nullptr;
  if (name == "zeroth_moment")
    get = get_zeroth;
  else if (name == "Vx_moment")
    get = get_Vx;
  else if (name == "Vy_moment")
    get = get_Vy;
  else if (name == "Vz_moment")
    get = get_Vz;
  else if (name == "Vr_moment")
    get = get_Vr;
  else if (name == "Vphi_moment")
    get = get_Vphi;
  else if (name == "mVxVx_moment")
    get = get_mVxVx;
  else if (name == "mVxVy_moment")
    get = get_mVxVy;
  else if (name == "mVxVz_moment")
    get = get_mVxVz;
  else if (name == "mVyVy_moment")
    get = get_mVyVy;
  else if (name == "mVyVz_moment")
    get = get_mVyVz;
  else if (name == "mVzVz_moment")
    get = get_mVzVz;
  else if (name == "mVrVr_moment")
    get = get_mVrVr;
  else if (name == "mVrVphi_moment")
    get = get_mVrVphi;
  else if (name == "mVphiVphi_moment")
    get = get_mVphiVphi;
  else
    throw std::runtime_error("Unkown moment name!");
  return Moment(particles, get);
}
