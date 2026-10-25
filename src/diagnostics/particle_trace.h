#ifndef SRC_DIAGNOSTICS_PARTICLE_TRACE_H
#define SRC_DIAGNOSTICS_PARTICLE_TRACE_H

#include "src/diagnostics/table_diagnostic.h"

/**
 * @brief Writes the time series of a single traced point. Supported point
 * types are `Point` (coordinate and momentum) and `PointByField` (guiding
 * center, drift-kinetic). The point is requested from `get_point` on each
 * diagnosed step, `nullptr` means that nothing is written.
 */
template<class PointT>
class ParticleTrace : public TableDiagnostic {
public:
  using Getter = std::function<const PointT*()>;

  ParticleTrace(const std::string& filename, Getter get_point, PetscInt skip);

  PetscErrorCode add_columns(PetscInt t) override;

private:
  Getter get_point;
  PetscInt skip;
};

#endif  // SRC_DIAGNOSTICS_PARTICLE_TRACE_H
