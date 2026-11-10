#include "src/utils/utils.h"

namespace MPIUtils {

PetscErrorCode log_statistics(std::string prefix, PetscInt agg, MPI_Comm comm)
{
  PetscFunctionBeginUser;
  PetscInt tot, min, max;
  PetscReal rat;

  PetscCallMPI(MPI_Allreduce(&agg, &tot, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&agg, &min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&agg, &max, 1, MPIU_INT, MPI_MAX, comm));
  rat = min > 0 ? (PetscReal)max / min : -1.0;

  LOG("{}total: {}, min: {}, max: {}, ratio: {:4.3f}", prefix, tot, min, max, rat);
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace MPIUtils
