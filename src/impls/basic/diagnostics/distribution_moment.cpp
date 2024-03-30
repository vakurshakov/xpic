#include "distribution_moment.h"

#include "src/utils/utils.h"
#include "src/vectors/vector_classes.h"

namespace basic {

Distribution_moment::Distribution_moment(
  MPI_Comm comm, const std::string& result_directory,
  const DM& da, const Particles& particles, Moment_up moment)
  : interfaces::Diagnostic(result_directory),
    da_(da), particles_(particles), moment_(std::move(moment)), comm_(comm) {}

Distribution_moment::~Distribution_moment() {
  PetscFunctionBeginUser;
  PetscCallVoid(DMDestroy(&da_));
  PetscCallVoid(VecDestroy(&local_));
  PetscCallVoid(VecDestroy(&global_));
  PetscFunctionReturnVoid();
}

PetscErrorCode Distribution_moment::set_diagnosed_region(const Region& region) {
  PetscFunctionBeginUser;

  region_ = region;
  PetscCall(setup_da());

  /// @todo Replace it with Field_view::set_diagnosed_region();
  ///
  Vector3<PetscInt> l_start, g_start = region.start;
  Vector3<PetscInt> m_size, f_size = region.size;
  PetscCall(DMDAGetCorners(da_, REP3_A(&l_start), REP3_A(&m_size)));

  l_start.to_petsc_order();
  g_start.to_petsc_order();
  m_size.to_petsc_order();
  f_size.to_petsc_order();

  Vector3<PetscInt> m_start = max(g_start, l_start);
  Vector3<PetscInt> l_size = min(g_start + f_size, l_start + m_size) - m_start;
  Vector3<PetscInt> f_start = m_start;

  f_start -= g_start;  // file start is in global coordinates, but we remove offset
  m_start -= l_start;  // memory start is in local coordinates

  PetscCall(file_.set_memview_subarray(3, m_size, l_size, m_start));
  PetscCall(file_.set_fileview_subarray(3, f_size, l_size, f_start));
  ///

  // Later we'll use `Node` structure that uses shifted coordinates
  region_.start -= shape_radius;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Distribution_moment::setup_da() {
  PetscFunctionBeginUser;
  Vector3<PetscInt> g_start = region_.start;
  Vector3<PetscInt> g_size = region_.size;
  Vector3<PetscInt> g_end = g_start + g_size;

  PetscInt dim, s;
  DMDAStencilType st;
  Vector3<PetscInt> size;
  Vector3<PetscInt> proc;
  Vector3<DMBoundaryType> bound;
  PetscCall(DMDAGetInfo(da_, &dim, REP3_A(&size), REP3_A(&proc), nullptr, &s, REP3_A(&bound), &st));

  Vector3<const PetscInt*> ownership;
  PetscCall(DMDAGetOwnershipRanges(da_, REP3_A(&ownership)));

  Vector3<PetscInt> l_proc;
  Vector3<DMBoundaryType> l_bound = DM_BOUNDARY_GHOSTED;
  Vector3<std::vector<PetscInt>> l_ownership;

  // Collecting number of processes and ownership ranges using global DMDA
  for (PetscInt i = 0; i < dim; ++i) {
    PetscInt start = 0;
    PetscInt end = ownership[i][0];

    for (PetscInt s = 0; s < proc[i]; ++s) {
      if (start < g_end[i] && end > g_start[i]) {
        l_proc[i]++;
        l_ownership[i].emplace_back(std::min(g_end[i], end) - std::max(g_start[i], start));
      }
      start += ownership[i][s];
      end += start;
    }

    // Mimic global boundaries, if we touch them
    if (g_size[i] == size[i]) {
      l_bound[i] = bound[i];
    }
  }

  PetscCall(DMDACreate3d(comm_, REP3_A(l_bound), st, REP3_A(g_size), REP3_A(l_proc), 1, s, l_ownership[X].data(), l_ownership[Y].data(), l_ownership[Z].data(), &da_));
  PetscCall(DMDASetOffset(da_, REP3_A(g_start), 0, 0, 0));
  PetscCall(DMSetUp(da_));
  PetscCall(DMCreateLocalVector(da_, &local_));
  PetscCall(DMCreateGlobalVector(da_, &global_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Distribution_moment::diagnose(timestep_t t) {
  if (t % diagnose_period != 0)
    PetscFunctionReturn(PETSC_SUCCESS);
  PetscFunctionBeginUser;

  PetscCall(collect());

  /// @todo Replace with Field_view::diagnose(t)
  ///
  int time_width = std::to_string(geom_nt).size();
  std::stringstream ss;
  ss << std::setw(time_width) << std::setfill('0') << t;
  PetscCall(file_.open(comm_, result_directory_, ss.str()));

  const PetscReal *arr;
  PetscCall(VecGetArrayRead(global_, &arr));

  Vector3<PetscInt> size;
  PetscCall(DMDAGetCorners(da_, REP3(nullptr), REP3_A(&size)));

  PetscCall(file_.write_floats(arr, (size[X] * size[Y] * size[Z])));
  PetscCall(file_.close());
  ///
  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @todo Communication is needed to prevent data losses. Moreover, the type of
 * data exchanges is dictated by the coordinates of the moment (`Projector`).
 * For example, if we collect particles density on (x, y, z) coordinates, we
 * need to exchange the values that are placed in ghost cells. But if we are
 * collecting velocity distribution on (Vx, Vy, Vz) coordinates, the equivalent
 * of `MPI_Allreduce()` operation is needed.
 */
PetscErrorCode Distribution_moment::collect() {
  PetscFunctionBeginUser;

  PetscCall(VecSet(local_, 0.0));
  PetscCall(VecSet(global_, 0.0));

  PetscReal ***arr;
  PetscCall(DMDAVecGetArrayWrite(da_, local_, &arr));

  #pragma omp parallel for
  for (const Point& point : particles_.get_points()) {
    Node node(point.r);

    /// @todo This can be moved into utilities
    bool in_bounds =
      (region_.start[X] <= node.g[X] && node.g[X] <= region_.start[X] + region_.size[X]) &&
      (region_.start[Y] <= node.g[Y] && node.g[Y] <= region_.start[Y] + region_.size[Y]) &&
      (region_.start[Z] <= node.g[Z] && node.g[Z] <= region_.start[Z] + region_.size[Z]);

    if (!in_bounds)
      continue;

    PetscReal n = particles_.density(point) / particles_.particles_number(point);

    PetscInt g_x, g_y, g_z;

    for (PetscInt z = 0; z < shape_width; ++z) {
    for (PetscInt y = 0; y < shape_width; ++y) {
    for (PetscInt x = 0; x < shape_width; ++x) {
      g_x = node.g[X] + x;
      g_y = node.g[Y] + y;
      g_z = node.g[Z] + z;

      #pragma omp atomic update
      arr[g_z][g_y][g_x] +=
        moment_->get(particles_, point) * n *
          shape_function(node.r[X] - g_x, X) *
          shape_function(node.r[Y] - g_y, Y) *
          shape_function(node.r[Z] - g_z, Z);
    }}}
  }
  PetscCall(DMDAVecRestoreArrayWrite(da_, local_, &arr));
  PetscCall(DMLocalToGlobal(da_, local_, ADD_VALUES, global_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/// @todo Maybe it would be better to move getters into *.h file and set them in the builder.
inline PetscReal get_zeroth(const Particles&, const Point&) {
  return 1.0;
}

inline PetscReal get_Vx(const Particles& particles, const Point& point) {
  return particles.velocity(point).x();
}

inline PetscReal get_Vy(const Particles& particles, const Point& point) {
  return particles.velocity(point).y();
}

inline PetscReal get_Vz(const Particles& particles, const Point& point) {
  return particles.velocity(point).z();
}

inline PetscReal get_mVxVx(const Particles& particles, const Point& point) {
  return particles.mass(point) * get_Vx(particles, point) * get_Vx(particles, point);
}

inline PetscReal get_mVxVy(const Particles& particles, const Point& point) {
  return particles.mass(point) * get_Vx(particles, point) * get_Vy(particles, point);
}

inline PetscReal get_mVxVz(const Particles& particles, const Point& point) {
  return particles.mass(point) * get_Vx(particles, point) * get_Vz(particles, point);
}

inline PetscReal get_mVyVy(const Particles& particles, const Point& point) {
  return particles.mass(point) * get_Vy(particles, point) * get_Vy(particles, point);
}

inline PetscReal get_mVyVz(const Particles& particles, const Point& point) {
  return particles.mass(point) * get_Vy(particles, point) * get_Vz(particles, point);
}

inline PetscReal get_mVzVz(const Particles& particles, const Point& point) {
  return particles.mass(point) * get_Vz(particles, point) * get_Vz(particles, point);
}

inline PetscReal get_Vr(const Particles& particles, const Point& point) {
  PetscReal x = point.x() - 0.5 * geom_x;
  PetscReal y = point.y() - 0.5 * geom_y;
  PetscReal r = sqrt(x * x + y * y);

  // Particles close to r=0 are not taken into account
  if (std::isinf(1.0 / r))
    return 0.0;

  return (+x * get_Vx(particles, point) + y * get_Vy(particles, point)) / r;
}

inline PetscReal get_Vphi(const Particles& particles, const Point& point) {
  PetscReal x = point.x() - 0.5 * geom_x;
  PetscReal y = point.y() - 0.5 * geom_y;
  PetscReal r = sqrt(x * x + y * y);

  // Particles close to r=0 are not taken into account
  if (std::isinf(1.0 / r))
    return 0.0;

  return (-y * get_Vx(particles, point) + x * get_Vy(particles, point)) / r;
}

inline PetscReal get_mVrVr(const Particles& particles, const Point& point) {
  return particles.mass(point) * get_Vr(particles, point) * get_Vr(particles, point);
}

inline PetscReal get_mVrVphi(const Particles& particles, const Point& point) {
  return particles.mass(point) * get_Vr(particles, point) * get_Vphi(particles, point);
}

inline PetscReal get_mVphiVphi(const Particles& particles, const Point& point) {
  return particles.mass(point) * get_Vphi(particles, point) * get_Vphi(particles, point);
}

Moment::Moment(const Particles& particles, const std::string& name) : particles_(particles) {
  if (name == "zeroth_moment") {
    get = get_zeroth;
  }
  else if (name == "Vx_moment") {
    get = get_Vx;
  }
  else if (name == "Vy_moment") {
    get = get_Vy;
  }
  else if (name == "Vz_moment") {
    get = get_Vz;
  }
  else if (name == "mVxVx_moment") {
    get = get_mVxVx;
  }
  else if (name == "mVxVy_moment") {
    get = get_mVxVy;
  }
  else if (name == "mVxVz_moment") {
    get = get_mVxVz;
  }
  else if (name == "mVyVy_moment") {
    get = get_mVyVy;
  }
  else if (name == "mVyVz_moment") {
    get = get_mVyVz;
  }
  else if (name == "mVzVz_moment") {
    get = get_mVzVz;
  }
  else if (name == "Vr_moment") {
    get = get_Vr;
  }
  else if (name == "Vphi_moment") {
    get = get_Vphi;
  }
  else if (name == "mVrVr_moment") {
    get = get_mVrVr;
  }
  else if (name == "mVrVphi_moment") {
    get = get_mVrVphi;
  }
  else if (name == "mVphiVphi_moment") {
    get = get_mVphiVphi;
  }
}

}
