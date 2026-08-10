#include "src/impls/drift_kinetic/segments.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using drift_kinetic::CellSplitMode;
using drift_kinetic::DriftKineticSegment;

void require(bool condition, const std::string& message)
{
  if (!condition)
    throw std::runtime_error(message);
}

DriftKineticSegment make_track(const Vector3R& start, const Vector3R& end)
{
  DriftKineticSegment track{};
  track.Rs0 = start;
  track.Rsn = end;
  track.dRs = end - start;
  track.Rsmid = 0.5 * (start + end);
  track.dRs_len = track.dRs.length();
  return track;
}

void require_inside_cell_center_box(
  const std::vector<DriftKineticSegment>& segments)
{
  for (const DriftKineticSegment& segment : segments) {
    for (Axis axis : {X, Y, Z}) {
      const PetscReal lower = -0.5;
      const PetscReal upper = static_cast<PetscReal>(Geom_n[axis]) - 0.5;
      require(segment.Rs0[axis] >= lower && segment.Rs0[axis] <= upper,
        "periodic segment start lies outside the cell-center box");
      require(segment.Rsn[axis] >= lower && segment.Rsn[axis] <= upper,
        "periodic segment end lies outside the cell-center box");
    }
  }
}

void require_preserved_length(const DriftKineticSegment& track,
  const std::vector<DriftKineticSegment>& segments)
{
  PetscReal split_length = 0.0;
  for (const DriftKineticSegment& segment : segments)
    split_length += segment.dRs_len;

  const PetscReal tolerance =
    256.0 * std::numeric_limits<PetscReal>::epsilon() *
    std::max(PetscReal{1.0}, track.dRs_len);
  require(std::abs(split_length - track.dRs_len) <= tolerance,
    "periodic splitting does not preserve trajectory length");
}

void test_log_roundoff_at_lower_bound()
{
  Geom_n[X] = 3;
  Geom_n[Y] = 3;
  Geom_n[Z] = 128;

  const DriftKineticSegment track = make_track(
    {0.25, -0.50000000000001421, -0.50000000000005684},
    {0.50, 0.25, 0.25});
  const auto segments = drift_kinetic::periodic_segments(
    track, CellSplitMode::cell_centers);

  require(!segments.empty(), "lower-bound track produced no segments");
  require(segments.front().Rs0[Y] == -0.5,
    "size-3 lower-bound roundoff was not snapped exactly");
  require(segments.front().Rs0[Z] == -0.5,
    "size-128 lower-bound roundoff was not snapped exactly");
  require_inside_cell_center_box(segments);
  require_preserved_length(track, segments);
}

void test_log_roundoff_at_upper_bound()
{
  Geom_n[X] = 3;
  Geom_n[Y] = 3;
  Geom_n[Z] = 128;

  const DriftKineticSegment track = make_track(
    {0.25, 2.50000000000001421, 127.50000000000005684},
    {0.0, 1.75, 126.75});
  const auto segments = drift_kinetic::periodic_segments(
    track, CellSplitMode::cell_centers);

  require(!segments.empty(), "upper-bound track produced no segments");
  require(segments.front().Rs0[Y] == 2.5,
    "size-3 upper-bound roundoff was not snapped exactly");
  require(segments.front().Rs0[Z] == 127.5,
    "size-128 upper-bound roundoff was not snapped exactly");
  require_inside_cell_center_box(segments);
  require_preserved_length(track, segments);
}

void test_periodic_crossing_preserves_trajectory()
{
  Geom_n[X] = 3;
  Geom_n[Y] = 3;
  Geom_n[Z] = 128;

  const DriftKineticSegment track = make_track(
    {2.25, 2.40, 127.35},
    {3.10, 3.20, 128.15});
  const auto segments = drift_kinetic::periodic_segments(
    track, CellSplitMode::cell_centers);

  require(segments.size() > 1,
    "boundary-crossing track was not split into periodic segments");
  require_inside_cell_center_box(segments);
  require_preserved_length(track, segments);
}

}  // namespace

int main()
{
  try {
    test_log_roundoff_at_lower_bound();
    test_log_roundoff_at_upper_bound();
    test_periodic_crossing_preserves_trajectory();
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
