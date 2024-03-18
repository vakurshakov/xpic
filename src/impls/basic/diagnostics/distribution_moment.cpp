#include "distribution_moment.h"

#include "src/utils/utils.h"
#include "src/vectors/vector_classes.h"

namespace basic {

Distribution_moment::Distribution_moment(
  const std::string& result_directory, const DM& da, const Particles& particles,
  Moment_up moment, Projector_up projector)
    : Diagnostic(result_directory), da_(da), particles_(particles),
      moment_(std::move(moment)), projector_(std::move(projector)) {}

PetscErrorCode Distribution_moment::set_diagnosed_region(const Region& region) {
  PetscFunctionBegin;
  Vector3<PetscInt> l_start, g_start = region.start;
  Vector3<PetscInt> l_size, g_size = region.size;
  PetscCall(DMDAGetCorners(da_, REP3_A(&l_start), REP3_A(&l_size)));

  l_start.to_petsc_order();
  l_size.to_petsc_order();

  g_start.to_petsc_order();
  g_size.to_petsc_order();

  l_start = max(g_start, l_start);
  l_size  = min(g_start + g_size, l_start + l_size) - l_start;
  g_start = l_start - g_start;

  region_.start = Vector3<PetscInt>{l_start};
  region_.size = Vector3<PetscInt>{l_size};
  region_.start.to_petsc_order();
  region_.size.to_petsc_order();
  region_.dp = region.dp;

  data_.resize(l_size[X] * l_size[Y] * l_size[Z]);

  PetscCall(file_.set_memview_subarray(3, l_size, l_size, Vector3<PetscInt>::null));
  PetscCall(file_.set_fileview_subarray(3, g_size, l_size, g_start));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Distribution_moment::diagnose(timestep_t t) {
  if (t % diagnose_period != 0)
    PetscFunctionReturn(PETSC_SUCCESS);
  PetscFunctionBeginUser;

  PetscCall(clear());
  PetscCall(collect());

  int time_width = std::to_string(geom_nt).size();
  std::stringstream ss;
  ss << std::setw(time_width) << std::setfill('0') << t;
  PetscCall(file_.open(PETSC_COMM_WORLD, result_directory_, ss.str()));

  const Vector3<PetscInt> size = region_.size;
  PetscCall(file_.write_floats(data_.data(), (size[X] * size[Y] * size[Z])));
  PetscCall(file_.close());

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Distribution_moment::collect() {
  PetscFunctionBeginUser;
  const PetscReal reg_dx = region_.dp[X];
  const PetscReal reg_dy = region_.dp[Y];
  const PetscReal reg_dz = region_.dp[Z];

  #pragma omp parallel for
  for (const Point& point : particles_.get_points()) {
    /// @todo It is a `Node` structure from `basic::Particles` except for region_.dp
    PetscReal p_rx = projector_->get_x(particles_, point) / reg_dx;
    PetscReal p_ry = projector_->get_y(particles_, point) / reg_dy;
    PetscReal p_rz = projector_->get_z(particles_, point) / reg_dz;

    PetscInt p_gx = ROUND(p_rx) - shape_radius;
    PetscInt p_gy = ROUND(p_ry) - shape_radius;
    PetscInt p_gz = ROUND(p_rz) - shape_radius;

    /// @todo Here we can use std::min()
    for (PetscInt g_z = p_gz; g_z < p_gz + shape_width; ++g_z) {
    for (PetscInt g_y = p_gy; g_y < p_gy + shape_width; ++g_y) {
    for (PetscInt g_x = p_gx; g_x < p_gx + shape_width; ++g_x) {
      bool in_bounds =
        (region_.start[X] <= g_x && g_x < region_.start[X] + region_.size[X]) &&
        (region_.start[Y] <= g_y && g_y < region_.start[Y] + region_.size[Y]) &&
        (region_.start[Z] <= g_z && g_z < region_.start[Z] + region_.size[Z]);

      if (!in_bounds)
        continue;

      PetscReal n = particles_.density(point) / particles_.particles_number(point);

      #pragma omp atomic update
      data_[index(g_x, g_y, g_z)] +=
        moment_->get(particles_, point) * n *
        shape_function(p_rx - g_x, X) *
        shape_function(p_ry - g_y, Y) *
        shape_function(p_rz - g_z, Z);
    }}}
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Distribution_moment::clear() {
  PetscFunctionBeginUser;
  #pragma omp parallel for
  for (PetscInt z = 0; z < region_.size[Z]; ++z) {
  for (PetscInt y = 0; y < region_.size[Y]; ++y) {
  for (PetscInt x = 0; x < region_.size[X]; ++x) {
    data_[index(x, y, z)] = 0.0;
  }}}
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


inline PetscReal project_to_x(const Particles&, const Point& point) {
  return point.x();
}

inline PetscReal project_to_y(const Particles&, const Point& point) {
  return point.y();
}

inline PetscReal project_to_z(const Particles&, const Point& point) {
  return point.z();
}

Projector::Projector(const Particles& particles, const std::string& axes_names) : particles_(particles) {
  if (axes_names == "(x_y_z)") {
    get_x = project_to_x;
    get_y = project_to_y;
    get_z = project_to_z;
  }
  else if (axes_names == "(Vx_Vy_Vz)") {
    get_x = get_Vx;
    get_y = get_Vy;
    get_z = get_Vz;
  }
  else if (axes_names == "(Vr_Vphi_Vz)") {
    get_x = get_Vr;
    get_y = get_Vphi;
    get_z = get_Vz;
  }
}

}
