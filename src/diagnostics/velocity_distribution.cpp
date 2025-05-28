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
  PetscInt m_start[2] = {vstart[Y], vstart[X]};
  PetscInt m_size[2] = {vsize[Y], vsize[X]};

  PetscInt l_start[2];
  PetscInt l_size[2];
  PetscCall(DMDAGetCorners(da_, REP2_AP(&l_start), nullptr, REP2_AP(&l_size), nullptr));

  l_start[X] -= m_start[X];
  l_start[Y] -= m_start[Y];

  PetscCall(file_.set_memview_subarray(2, m_size, l_size, l_start));
  PetscCall(file_.set_fileview_subarray(2, m_size, l_size, l_start));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/// @todo Test-condition is required for spatial coordinates
/// @todo Projector here is required in case we would need to
/// map `point.p` onto cylinder or field-aligned coordinates.
/// Without it, we wouldn't be able to remap components later.
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

    for (auto&& point : particles_.storage[g]) {
      vg[X] = ROUND_STEP(point.px(), vreg.dvx);
      vg[Y] = ROUND_STEP(point.py(), vreg.dvy);
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
