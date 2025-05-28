#include "velocity_distribution.h"

#include "src/utils/geometries.h"
#include "src/utils/utils.h"
#include "src/utils/vector_utils.h"


std::unique_ptr<VelocityDistribution> VelocityDistribution::create(
  const std::string& out_dir, const interfaces::Particles& particles,
  const Region& xreg_aabb, const VelocityRegion& vreg)
{
  PetscFunctionBeginUser;
  // Communicator is based on axis-aligned bounding box of space integration domain
  MPI_Comm newcomm;
  PetscCallAbort(PETSC_COMM_WORLD, get_local_communicator(particles.world.da, xreg_aabb, &newcomm));
  if (newcomm == MPI_COMM_NULL)
    PetscFunctionReturn(nullptr);

  auto* diagnostic = new VelocityDistribution(out_dir, particles, newcomm);
  PetscCallAbort(PETSC_COMM_WORLD, diagnostic->set_regions(xreg_aabb, vreg));
  PetscFunctionReturn(std::unique_ptr<VelocityDistribution>(diagnostic));
}

VelocityDistribution::VelocityDistribution(const std::string& out_dir,
  const interfaces::Particles& particles, MPI_Comm newcomm)
  : DistributionMoment(out_dir, particles, nullptr, newcomm)
{
}

PetscErrorCode VelocityDistribution::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(ISDestroy(&is));
  PetscCall(VecScatterDestroy(&ctx));
  PetscCall(VecDestroy(&local_));
  PetscCall(VecDestroy(&field_));
  PetscCall(DMDestroy(&da_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VelocityDistribution::set_regions(
  const Region& xreg_aabb, const VelocityRegion& vreg)
{
  PetscFunctionBeginUser;
  this->xreg_aabb = xreg_aabb;
  this->vreg = vreg;

  xstart = vector_cast(xreg_aabb.start);
  xsize = vector_cast(xreg_aabb.size);

  vstart = Vector3I{
    ROUND_STEP(vreg.vx_min, vreg.dvx),
    ROUND_STEP(vreg.vx_min, vreg.dvx),
    0,
  };

  vsize = Vector3I{
    ROUND_STEP(vreg.vx_max - vreg.vx_min, vreg.dvx),
    ROUND_STEP(vreg.vx_max - vreg.vx_min, vreg.dvx),
    1,
  };

  // da in coordinate space
  global_da_ = da_;

  const PetscInt size = vsize.elements_product();

  // da should be re-created (in velocity space) to be used in FieldView and set up the field, corners
  PetscCall(DMDACreate2d(comm_, REP2(DM_BOUNDARY_NONE), DMDA_STENCIL_BOX, REP2_A(vsize), REP2(PETSC_DECIDE), 1, 0, REP2(nullptr), &da_));
  PetscCall(DMDASetOffset(da_, REP3_A(vstart), REP3(0)));
  PetscCall(DMSetUp(da_));

  // field is mpi (when needed), local vector is sequential
  PetscCall(DMCreateGlobalVector(da_, &field_));
  PetscCall(VecCreate(PETSC_COMM_SELF, &local_));
  PetscCall(VecSetSizes(local_, size, size));
  PetscCall(VecSetType(local_, VECSEQ));

  PetscCall(ISCreateStride(comm_, size, 0, 1, &is));
  PetscCall(VecScatterCreate(local_, is, field_, is, &ctx));

  PetscCall(set_data_views({}));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// `FieldView::set_data_views()` is simplified because the whole `field_` will be written
PetscErrorCode VelocityDistribution::set_data_views(const Region& /* reg */)
{
  PetscInt m_start[2] = {0, 0};
  PetscInt g_start[2] = {vstart[Y], vstart[X]};
  PetscInt g_size[2] = {vsize[Y], vsize[X]};

  PetscInt l_start[2];
  PetscInt l_size[2];
  PetscCall(DMDAGetCorners(da_, REP2_AP(&l_start), nullptr, REP2_AP(&l_size), nullptr));

  l_start[X] -= g_start[X];
  l_start[Y] -= g_start[Y];

  PetscCall(file_.set_memview_subarray(2, l_size, l_size, m_start));
  PetscCall(file_.set_fileview_subarray(2, g_size, l_size, l_start));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode VelocityDistribution::collect()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(local_, 0.0));
  PetscCall(VecSet(field_, 0.0));

  PetscReal** arr;
  PetscCall(VecGetArray2dWrite(local_, REP2_A(vsize), REP2_A(vstart), &arr));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(global_da_, REP3_A(&start), REP3_A(&size)));

#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    Vector3I vg{
      start[X] + g % size[X],
      start[Y] + (g / size[X]) % size[Y],
      start[Z] + (g / size[X]) / size[Y],
    };

    if (!is_point_within_bounds(vg, xstart, xsize))
      continue;

    Vector3R r{
      (vg[X] + 0.5) * dx,
      (vg[Y] + 0.5) * dy,
      (vg[Z] + 0.5) * dz,
    };

    if (!within_geom(r))
      continue;

    for (auto&& point : particles_.storage[g]) {
      auto [vx, vy] = projector(point);

      vg[X] = ROUND_STEP(vx, vreg.dvx);
      vg[Y] = ROUND_STEP(vy, vreg.dvy);
      vg[Z] = 0;

      if (!is_point_within_bounds(vg, vstart, vsize))
        continue;

#pragma omp atomic update
      arr[vg[Y]][vg[X]] += particles_.n_Np(point);
    }
  }

  PetscCall(VecRestoreArray2dWrite(local_, REP2_A(vsize), REP2_A(vstart), &arr));
  PetscCall(VecScatterBegin(ctx, local_, field_, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx, local_, field_, ADD_VALUES, SCATTER_FORWARD));
  PetscFunctionReturn(PETSC_SUCCESS);
}


std::tuple<PetscReal, PetscReal> get_vx_vy(const Point& point)
{
  return {point.px(), point.py()};
}

std::tuple<PetscReal, PetscReal> get_vz_vxy(const Point& point)
{
  const Vector3R perp(point.px(), point.py(), 0);
  return {point.pz(), perp.length()};
}

std::tuple<PetscReal, PetscReal> get_vr_vphi(const Point& point)
{
  PetscReal x = point.x() - 0.5 * geom_x;
  PetscReal y = point.y() - 0.5 * geom_y;
  PetscReal r = std::hypot(x, y);

  auto&& v = point.p;

  // Particles close to r=0 are not taken into account
  if (std::isinf(1.0 / r))
    return {v[X], v[Y]};

  return {
    (+x * v[X] + y * v[Y]) / r,
    (-y * v[X] + x * v[Y]) / r,
  };
}

Projector projector_from_string(const std::string& name)
{
  if (name == "vx_vy")
    return get_vx_vy;
  if (name == "vz_vxy")
    return get_vz_vxy;
  if (name == "vr_vphi")
    return get_vr_vphi;

  throw std::runtime_error("Unkown projector name " + std::string(name));
}
