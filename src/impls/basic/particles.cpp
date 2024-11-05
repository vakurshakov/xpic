#include "particles.h"

#include "src/impls/basic/simulation.h"
#include "src/utils/simple_interpolation.h"

namespace basic {

Particles::Particles(Simulation& simulation, const Sort_parameters& parameters)
  : interfaces::Particles(simulation.world_, parameters), simulation_(simulation)
{
  PetscFunctionBeginUser;
  /// @note This local current is local to each particle!
  /// It's can be useful for diagnosing it.
  DM da = simulation_.world_.da;
  PetscCallVoid(DMCreateLocalVector(da, &local_J));
  PetscFunctionReturnVoid();
}

Particles::~Particles()
{
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&local_J));
  PetscFunctionReturnVoid();
}


PetscErrorCode Particles::push()
{
  PetscFunctionBeginUser;
  DM da = simulation_.world_.da;
  PetscCall(DMGetLocalVector(da, &local_E));
  PetscCall(DMGetLocalVector(da, &local_B));

  PetscCall(DMGlobalToLocal(da, simulation_.E_, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, simulation_.B_, INSERT_VALUES, local_B));
  PetscCall(VecSet(local_J, 0.0));

  PetscCall(DMDAVecGetArrayRead(da, local_E, &E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayWrite(da, local_J, &J));

#pragma omp for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    Vector3R point_E = 0.0;
    Vector3R point_B = 0.0;

    const Node node(it->r);

    static Shape shape[2];
#pragma omp threadprivate(shape)

    fill_shape(node.g, node.r, world_.shape_size, false, shape[0]);
    fill_shape(node.g, node.r, world_.shape_size, true, shape[1]);
    interpolate(node.g, shape[0], shape[1], point_E, point_B);

    push(point_E, point_B, *it);

    const Node new_node(it->r);

    fill_shape(new_node.g, node.r, world_.shape_size, false, shape[0]);
    fill_shape(new_node.g, new_node.r, world_.shape_size, false, shape[1]);
    decompose(new_node.g, shape[0], shape[1], *it);
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_J, &J));

  PetscCall(DMLocalToGlobal(da, local_J, ADD_VALUES, simulation_.J_));

  PetscCall(DMRestoreLocalVector(da, &local_E));
  PetscCall(DMRestoreLocalVector(da, &local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}


void Particles::interpolate(const Vector3I& p_g, Shape& no, Shape& sh,
  Vector3R& point_E, Vector3R& point_B) const
{
  Simple_interpolation interpolation(world_.shape_size, no, sh);
  interpolation.process(p_g, {{point_E, E}}, {{point_B, B}});
}


void Particles::push(
  const Vector3R& point_E, const Vector3R& point_B, Point& point) const
{
  PetscReal alpha = 0.5 * dt * charge(point);
  PetscReal m = mass(point);

  Vector3R& r = point.r;
  Vector3R& p = point.p;

  const Vector3R w = p + point_E * alpha;
  PetscReal energy = sqrt(m * m + w.dot(w));

  const Vector3R h = point_B * alpha / energy;
  const Vector3R s = h * 2.0 / (1.0 + h.dot(h));
  p = point_E * alpha + w * (1.0 - h.dot(s)) + w.cross(s) + h * (s.dot(w));

  energy = sqrt(m * m + p.dot(p));
  r += p * dt / energy;

  if (geom_nx == 1)
    r.x() = 0.5;
  if (geom_ny == 1)
    r.y() = 0.5;
  if (geom_nz == 1)
    r.z() = 0.5;
}


void Particles::decompose(
  const Vector3I& p_g, Shape& old_shape, Shape& new_shape, const Point& point)
{
  const PetscReal alpha =
    charge(point) * density(point) / particles_number(point) / (6.0 * dt);
  const PetscReal qx = alpha * dx;
  const PetscReal qy = alpha * dy;
  const PetscReal qz = alpha * dz;

  auto compute_Jx = [&](PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jx) {
    PetscInt i = ((z * shape_width + y) * shape_width + x);
    PetscInt j = (z * shape_width + y);

    PetscReal p_wx = -qx * (new_shape(i, X) - old_shape(i, X)) *
      (new_shape(i, Y) * (2.0 * new_shape(i, Z) + old_shape(i, Z)) +
        old_shape(i, Y) * (2.0 * old_shape(i, Z) + new_shape(i, Z)));

    temp_jx[j] = ((x > 0) * temp_jx[j]) + p_wx;
    return temp_jx[j];
  };

  auto compute_Jy = [&](PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jy) {
    PetscInt i = ((z * shape_width + y) * shape_width + x);
    PetscInt j = (z * shape_width + x);

    PetscReal p_wy = -qy * (new_shape(i, Y) - old_shape(i, Y)) *
      (new_shape(i, X) * (2.0 * new_shape(i, Z) + old_shape(i, Z)) +
        old_shape(i, X) * (2.0 * old_shape(i, Z) + new_shape(i, Z)));

    temp_jy[j] = ((y > 0) * temp_jy[j]) + p_wy;
    return temp_jy[j];
  };

  auto compute_Jz = [&](PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jz) {
    PetscInt i = ((z * shape_width + y) * shape_width + x);
    PetscInt j = (y * shape_width + x);

    PetscReal p_wz = -qz * (new_shape(i, Z) - old_shape(i, Z)) *
      (new_shape(i, Y) * (2.0 * new_shape(i, X) + old_shape(i, X)) +
        old_shape(i, Y) * (2.0 * old_shape(i, X) + new_shape(i, X)));

    temp_jz[j] = ((z > 0) * temp_jz[j]) + p_wz;
    return temp_jz[j];
  };

  decompose_dir(p_g, compute_Jx, X);
  decompose_dir(p_g, compute_Jy, Y);
  decompose_dir(p_g, compute_Jz, Z);
}

void Particles::decompose_dir(
  const Vector3I& p_g, const Compute_j& compute_j, Axis dir)
{
  static PetscReal temp_j[shape_width * shape_width];
#pragma omp threadprivate(temp_j)

  PetscInt g_x, g_y, g_z;

  // clang-format off
  for (PetscInt z = 0; z < world_.shape_size[Z]; ++z) {
  for (PetscInt y = 0; y < world_.shape_size[Y]; ++y) {
  for (PetscInt x = 0; x < world_.shape_size[X]; ++x) {
    g_x = p_g[X] + x;
    g_y = p_g[Y] + y;
    g_z = p_g[Z] + z;

    PetscReal p_j = compute_j(x, y, z, temp_j);

#pragma omp atomic update
    J[g_z][g_y][g_x][dir] += p_j;
  }}}
  // clang-format on
}

}  // namespace basic
