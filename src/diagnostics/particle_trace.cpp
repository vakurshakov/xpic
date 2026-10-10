#include "particle_trace.h"

template<class PointT>
ParticleTrace<PointT>::ParticleTrace(
  const std::string& filename, Getter get_point, PetscInt skip)
  : TableDiagnostic(filename), get_point(get_point), skip(skip)
{
}

template<class PointT>
PetscErrorCode ParticleTrace<PointT>::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  const PointT* point = get_point();
  if (t % skip != 0 || point == nullptr)
    PetscFunctionReturn(PETSC_SUCCESS);

  if constexpr (std::is_same_v<PointT, Point>) {
    add(20, "t_[1/wpe]", "{: .9e}", t * dt);
    add(20, "x_[c/wpe]", "{: .9e}", point->x());
    add(20, "y_[c/wpe]", "{: .9e}", point->y());
    add(20, "z_[c/wpe]", "{: .9e}", point->z());
    add(20, "vx_[c]", "{: .9e}", point->px());
    add(20, "vy_[c]", "{: .9e}", point->py());
    add(20, "vz_[c]", "{: .9e}", point->pz());
  }
  else if constexpr (std::is_same_v<PointT, PointByField>) {
    add(20, "t_[1/wpe]", "{: .15e}", t * dt);
    add(20, "x_[c/wpe]", "{: .15e}", point->x());
    add(20, "y_[c/wpe]", "{: .15e}", point->y());
    add(20, "z_[c/wpe]", "{: .15e}", point->z());
    add(20, "p_par_[mc]", "{: .15e}", point->p_par());
    add(20, "p_perp_[mc]", "{: .15e}", point->p_perp());
    add(20, "mu_p_[mc^2/B]", "{: .15e}", point->mu());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template class ParticleTrace<Point>;
template class ParticleTrace<PointByField>;
