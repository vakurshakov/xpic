#ifndef SRC_IMPLS_DRIFT_KINETIC_SEGMENTS_H
#define SRC_IMPLS_DRIFT_KINETIC_SEGMENTS_H

#include <vector>

#include <petscdm.h>

#include "src/utils/shape.h"
#include "src/utils/geometries.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace drift_kinetic {

enum class CellSplitMode {
  cell_edges,    // split at integer boundaries: 0, 1, 2, ...
  cell_centers,  // split at half-integer boundaries: 0.5, 1.5, 2.5, ...
};

struct DriftKineticSegment{
  Vector3R Rs0;
  Vector3R Rsn;
  Vector3R Rsmid;
  Vector3R dRs;
  PetscReal dRs_len;
};

DriftKineticSegment make_track(const Vector3R& R0, const Vector3R& Rn);
Vector3R make_mid(const DriftKineticSegment& segment);
Vector3R make_end(const DriftKineticSegment& segment);
Vector3R make_begin(const DriftKineticSegment& segment);

std::vector<DriftKineticSegment> periodic_segments(
  const DriftKineticSegment& track,
  CellSplitMode mode = CellSplitMode::cell_edges);

std::vector<DriftKineticSegment> cell_segments(
  const std::vector<DriftKineticSegment>& segments,
  CellSplitMode mode = CellSplitMode::cell_edges);

} //namespace drift_kinetic

#endif  // SRC_IMPLS_DRIFT_KINETIC_SEGMENTS_H
