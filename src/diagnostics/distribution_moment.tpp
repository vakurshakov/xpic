#ifndef SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_TPP
#define SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_TPP

#include "src/utils/shape.h"

template<typename ForEachParticle>
PetscErrorCode DistributionMoment::collect(ForEachParticle&& for_each_particle)
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(scratch, 0));
  PetscCall(VecSet(scratch_loc, 0));

  PetscReal**** arr;
  PetscCall(DMDAVecGetArrayDOFWrite(da_scratch, scratch_loc, &arr));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da_glob, REP3_A(&start), REP3_A(&size)));

  const PetscReal mpw = particles.parameters.mpw;

#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); g++) {
    auto deposit = [&](const Vector3R& r, const std::vector<PetscReal>& moments)
    {
      for_each_shape_node(r,
        [&](PetscInt x, PetscInt y, PetscInt z, PetscReal w)
        {
          for (std::size_t j = 0; j < moments.size(); j++) {
#pragma omp atomic update
            arr[z][y][x][j] += moments[j] * (w * mpw);
          }
        });
    };

    for_each_particle(g, deposit);
  }
  PetscCall(DMDAVecRestoreArrayDOFWrite(da_scratch, scratch_loc, &arr));
  PetscCall(DMLocalToGlobal(da_scratch, scratch_loc, ADD_VALUES, scratch));

  if (comm == MPI_COMM_NULL)
    PetscFunctionReturn(PETSC_SUCCESS);

  // The region DM mimics the world ownership, so every owned cell is copied locally
  const PetscInt dof = region.dof;
  const PetscReal**** src;
  PetscReal**** dst;
  PetscCall(DMDAVecGetArrayDOFRead(da_scratch, scratch, &src));
  PetscCall(DMDAVecGetArrayDOFWrite(da, field, &dst));

  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));
  for (PetscInt z = start[Z]; z < start[Z] + size[Z]; z++)
    for (PetscInt y = start[Y]; y < start[Y] + size[Y]; y++)
      for (PetscInt x = start[X]; x < start[X] + size[X]; x++)
        for (PetscInt j = 0; j < dof; j++)
          dst[z][y][x][j] = src[z][y][x][j];

  PetscCall(DMDAVecRestoreArrayDOFRead(da_scratch, scratch, &src));
  PetscCall(DMDAVecRestoreArrayDOFWrite(da, field, &dst));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif  // SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_TPP
