#ifndef SRC_IMPLS_DRIFT_KINETIC_SEGMENTS_H
#define SRC_IMPLS_DRIFT_KINETIC_SEGMENTS_H

#include <vector>

#include "src/utils/vector3.h"

/// @file segments.h
/// @brief Periodic trajectory wrapping and cell segmentation.

namespace drift_kinetic {

/// @brief Selects the cell partition and periodic box origin.
enum class CellSplitMode {
  cell_edges,    ///< Integer planes and a periodic box starting at zero.
  cell_centers,  ///< Half-integer planes and a box starting at minus one half.
};

/// @brief Straight track segment in grid coordinates.
struct DriftKineticSegment {
  Vector3R Rs0;       ///< Start point.
  Vector3R Rsn;       ///< End point.
  Vector3R Rsmid;     ///< Midpoint.
  Vector3R dRs;       ///< Displacement.
  PetscReal dRs_len;  ///< Length in grid coordinates.
};

/// @brief Converts physical endpoints into a consistent grid track.
DriftKineticSegment make_track(const Vector3R& R0, const Vector3R& Rn);
/// @brief Returns the physical end point.
Vector3R make_end(const DriftKineticSegment& segment);
/// @brief Returns the physical start point.
Vector3R make_begin(const DriftKineticSegment& segment);

/// @brief Splits a track at periodic boundaries and wraps its pieces.
std::vector<DriftKineticSegment> periodic_segments(
  const DriftKineticSegment& track,
  CellSplitMode mode = CellSplitMode::cell_edges);

/// @brief Splits consistent grid segments at the selected cell planes.
std::vector<DriftKineticSegment> cell_segments(
  const std::vector<DriftKineticSegment>& segments,
  CellSplitMode mode = CellSplitMode::cell_edges);

}  // namespace drift_kinetic

#endif  // SRC_IMPLS_DRIFT_KINETIC_SEGMENTS_H
