#ifndef SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_TPP
#define SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_TPP

#include "src/utils/geometries.h"
#include "src/utils/vector_utils.h"

template<typename ForEachParticle>
PetscErrorCode DistributionMoment::collect(ForEachParticle&& for_each_particle)
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

#pragma omp parallel for
  for (PetscInt g = 0; g < size.elements_product(); ++g) {
    Vector3I vg{
      start[X] + g % size[X],
      start[Y] + (g / size[X]) % size[Y],
      start[Z] + (g / size[X]) / size[Y],
    };

    if (!is_point_within_bounds(vg, gstart, gsize))
      continue;

    // The lambda must be created inside the parallel region to use a thread-local `shape`
    Shape shape;

    auto deposit = [&](const Vector3R& r, const std::vector<PetscReal>& moments)
    {
      shape.setup(r);
      auto msize = static_cast<PetscInt>(moments.size());

      for (PetscInt i = 0; i < shape.shm; ++i) {
        PetscInt g_x = shape.start[X] + i % shape.shw;
        PetscInt g_y = shape.start[Y] + (i / shape.shw) % shape.shw;
        PetscInt g_z = shape.start[Z] + (i / shape.shw) / shape.shw;

        PetscReal si = shape.cache[i] * particles.parameters.mpw;

        for (PetscInt j = 0; j < msize; ++j) {
          PetscReal mj = moments[j] * si;

#pragma omp atomic update
          arr[g_z][g_y][g_x][j] += mj;
        }
      }
    };

    for_each_particle(g, deposit);
  }
  PetscCall(DMDAVecRestoreArrayDOFWrite(da, field_loc, &arr));
  PetscCall(DMLocalToGlobal(da, field_loc, ADD_VALUES, field));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif  // SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_TPP
