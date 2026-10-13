#ifndef SRC_IMPLS_DRIFT_KINETIC_SEGMENTS_H
#define SRC_IMPLS_DRIFT_KINETIC_SEGMENTS_H

#include "src/utils/vector3.h"

namespace drift_kinetic {

/// @brief Selects the cell partition and periodic box origin.
enum class CellSplitMode {
  edges,    // Integer planes and a periodic box starting at zero.
  centers,  // Half-integer planes and a box starting at minus one half.
};

/// @brief Straight track segment in grid coordinates.
struct DriftKineticSegment {
  Vector3R Rs0;       // Start point.
  Vector3R Rsn;       // End point.
  Vector3R Rsh;       // Midpoint.
  Vector3R dRs;       // Displacement.
  PetscReal dRs_len;  // Length in grid coordinates.
};

/// @brief Converts physical endpoints into a consistent grid track.
DriftKineticSegment make_track(const Vector3R& R0, const Vector3R& Rn);

Vector3R make_begin(const DriftKineticSegment& segment);
Vector3R make_end(const DriftKineticSegment& segment);

/// @brief Splits a track at periodic boundaries and wraps its pieces.
std::vector<DriftKineticSegment> periodic_segments(
  const DriftKineticSegment& track, CellSplitMode mode);

/// @brief Splits consistent grid segments at the selected cell planes.
std::vector<DriftKineticSegment> cell_segments(
  const std::vector<DriftKineticSegment>& segments, CellSplitMode mode);

}  // namespace drift_kinetic

#endif  // SRC_IMPLS_DRIFT_KINETIC_SEGMENTS_H
