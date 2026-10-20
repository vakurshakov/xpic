#include "segments.h"

#include "src/utils/shape.h"

namespace drift_kinetic {
namespace {

DriftKineticSegment make_segment(const Vector3R& Rs0, const Vector3R& Rsn)
{
  return DriftKineticSegment{
    .Rs0 = Rs0,
    .Rsn = Rsn,
    .Rsh = 0.5 * (Rs0 + Rsn),
    .dRs = Rsn - Rs0,
    .dRs_len = (Rsn - Rs0).length(),
  };
}

}  // namespace

DriftKineticSegment make_track(const Vector3R& R0, const Vector3R& Rn)
{
  return make_segment(::Shape::make_r(R0), ::Shape::make_r(Rn));
}

Vector3R make_begin(const DriftKineticSegment& segment)
{
  return {
    segment.Rs0[X] * dx,
    segment.Rs0[Y] * dy,
    segment.Rs0[Z] * dz,
  };
}

Vector3R make_end(const DriftKineticSegment& segment)
{
  return {
    segment.Rsn[X] * dx,
    segment.Rsn[Y] * dy,
    segment.Rsn[Z] * dz,
  };
}

// Cell planes sit at x = n - offset, cell n is [n - offset, n + 1 - offset),
// so its index is floor(x + offset). Planes strictly crossed by the track are
// n = min(c0, c1) + 1 .. max(c0, c1); a plane touched by an endpoint, or one
// misjudged by rounding, yields t in {0, 1} after the clamp and is merged by
// the exact unique. Near-ties of different axes leave slivers, harmless since
// the shape functions are continuous across cell faces.
//
// Each piece takes the cell of its midpoint, wrapped into [0, Geom_n) in
// integers, and is shifted by the same whole number of periods. The cell is
// stored, not rederived from the shifted coordinates, which may round onto
// the opposite box face.
std::vector<DriftKineticSegment> cell_segments(
  const DriftKineticSegment& track, CellSplitMode mode)
{
  PetscReal offset = ((mode == CellSplitMode::centers) ? 0.5 : 0);

  std::vector<PetscReal> t_values{0, 1};
  for (Axis axis : {X, Y, Z}) {
    auto c0 = (PetscInt)std::floor(track.Rs0[axis] + offset);
    auto c1 = (PetscInt)std::floor(track.Rsn[axis] + offset);

    for (PetscInt n = std::min(c0, c1) + 1; n <= std::max(c0, c1); n++) {
      PetscReal t = ((PetscReal)n - offset - track.Rs0[axis]) / track.dRs[axis];
      t_values.push_back(std::clamp(t, PetscReal{0}, PetscReal{1}));
    }
  }
  std::sort(t_values.begin(), t_values.end());
  t_values.erase(std::unique(t_values.begin(), t_values.end()), t_values.end());

  std::vector<DriftKineticSegment> segments;
  segments.reserve(t_values.size() - 1);

  Vector3R Rsn = track.Rs0;
  for (std::size_t i = 1; i < t_values.size(); i++) {
    Vector3R Rs0 = Rsn;
    Rsn = (i + 1 < t_values.size()) ? track.Rs0 + track.dRs * t_values[i]
                                    : track.Rsn;

    Vector3R shifted_Rs0 = Rs0;
    Vector3R shifted_Rsn = Rsn;
    Vector3I cell;
    for (Axis axis : {X, Y, Z}) {
      auto c = (PetscInt)std::floor(0.5 * (Rs0[axis] + Rsn[axis]) + offset);
      cell[axis] = ((c % Geom_n[axis]) + Geom_n[axis]) % Geom_n[axis];

      auto shift = (PetscReal)(c - cell[axis]);
      shifted_Rs0[axis] -= shift;
      shifted_Rsn[axis] -= shift;
    }

    auto segment = make_segment(shifted_Rs0, shifted_Rsn);
    segment.cell = cell;
    segments.push_back(segment);
  }
  return segments;
}

}  // namespace drift_kinetic
