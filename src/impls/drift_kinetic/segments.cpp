#include "segments.h"

#include "src/utils/shape.h"

namespace drift_kinetic {
namespace {

constexpr PetscReal split_eps = 1e-14;
constexpr PetscReal split_tie_eps = 1e-13;

struct PeriodicBox {
  PetscReal lower;
  PetscReal upper;
  PetscReal length;
  PetscReal tolerance;
};

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

PeriodicBox make_periodic_box(PetscInt axis, CellSplitMode mode)
{
  PetscReal length = (PetscReal)Geom_n[axis];
  PetscCheckAbort(length > split_eps, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
    "Invalid periodic length on axis %d: Geom_n[%d]=%.16e", axis, axis, length);

  PetscReal lower = ((mode == CellSplitMode::centers) ? -0.5 : 0);

  // The positive integer length bounds both box coordinates in either mode.
  PetscReal tolerance = 64.0 * std::numeric_limits<PetscReal>::epsilon() * length;

  return PeriodicBox{
    lower,
    lower + length,
    length,
    tolerance,
  };
}

PetscReal clamp_periodic_coordinate(PetscReal value, const PeriodicBox& box)
{
  if (std::abs(value - box.lower) <= box.tolerance)
    return box.lower;
  if (std::abs(value - box.upper) <= box.tolerance)
    return box.upper;
  return value;
}

void sort_unique_t_values(std::vector<PetscReal>& t_values)
{
  std::sort(t_values.begin(), t_values.end());

  std::size_t kept = 0;
  for (PetscReal t : t_values) {
    if (kept == 0 || std::abs(t - t_values[kept - 1]) > split_tie_eps)
      t_values[kept++] = t;
  }
  t_values.resize(kept);
}

// (1) Shifts the whole segment by an integer number of box lengths, so that its
// `start` lies in [lower, upper]. The `end` may still leave the box.
//
//   lower                 upper
//     |<------ length ------>|
//     |                      |
//     |   s---->e            |          inside: unchanged
//     |                      |
//  s--+-->e                  |          s < lower: shift by +length * k
//     |                      |
//     |                      | s--->e   s > upper: shift by -length * k
//     |                      |
//     |          s-----------+-->e      s stays, e is wrapped later by splits
//     |                      |
//
// (2) A `start` exactly on a face with an outward direction is moved to the
// opposite face, so that the first sub-segment always points inward:
//
//   e<--s|                  |      =>      |                  |e<--s
//        lower                                              upper
//   (dir < 0 and start == lower: start += length, end += length)
//
void canonicalize_to_periodic_box(
  PetscReal& start, PetscReal& end, const PeriodicBox& box)
{
  PetscReal image_shift = std::floor((start - box.lower) / box.length);  // (1)
  start -= image_shift * box.length;
  end -= image_shift * box.length;

  if (start < box.lower) {
    start += box.length;
    end += box.length;
  }
  else if (start > box.upper) {
    start -= box.length;
    end -= box.length;
  }

  start = clamp_periodic_coordinate(start, box);

  PetscReal dir = end - start;  // (2)
  if (dir < -split_eps && start == box.lower) {
    start += box.length;
    end += box.length;
  }
  else if (dir > split_eps && start == box.upper) {
    start -= box.length;
    end -= box.length;
  }

  start = clamp_periodic_coordinate(start, box);
}

struct PeriodicSplitEvents {
  std::vector<PetscReal> t_values;
  std::vector<PetscReal> wrap_t_values[3];
  PetscReal wrap_delta[3]{0, 0, 0};
};

// Segment parameters t in (0, 1), at which the segment crosses a box face
// lower + k * length. Per axis, with dir = end - start:
//
//   lower + 0L      lower + 1L      lower + 2L     (faces k = 0, 1, 2)
//     |               |               |
//     |     s---------+-----------e   |     dir > 0: k = k_begin..k_end
//     |               ^               |     t = (boundary - start) / dir
//     |               t               |     here only k = 1 is crossed
//
// After each crossing the coordinate is wrapped by wrap_delta (-length for
// dir > 0, +length for dir < 0). t_values collects crossings over all axes,
// wrap_t_values keeps them per axis to count the wraps.
//
PeriodicSplitEvents collect_periodic_split_params(
  const Vector3R& start, const Vector3R& end, const PeriodicBox (&boxes)[3])
{
  PeriodicSplitEvents events;

  for (Axis axis : {X, Y, Z}) {
    const auto& box = boxes[axis];
    PetscReal dir = end[axis] - start[axis];
    PetscReal start_shifted = start[axis] - box.lower;
    PetscReal end_shifted = end[axis] - box.lower;

    if (std::abs(dir) <= split_eps)
      continue;

    if (dir > 0) {
      events.wrap_delta[axis] = -box.length;

      auto k_begin = (PetscInt)std::floor(start_shifted / box.length) + 1;
      auto k_end = (PetscInt)std::floor((end_shifted - split_eps) / box.length);

      for (PetscInt k = k_begin; k <= k_end; k++) {
        PetscReal boundary = box.lower + (PetscReal)k * box.length;
        PetscReal t = (boundary - start[axis]) / dir;
        if (t > split_eps && t < 1.0 - split_eps) {
          events.t_values.push_back(t);
          events.wrap_t_values[axis].push_back(t);
        }
      }
    }
    else {
      events.wrap_delta[axis] = box.length;

      auto k_begin = (PetscInt)std::ceil(start_shifted / box.length) - 1;
      auto k_end = (PetscInt)std::ceil((end_shifted + split_eps) / box.length);

      for (PetscInt k = k_begin; k >= k_end; k--) {
        PetscReal boundary = box.lower + (PetscReal)k * box.length;
        PetscReal t = (boundary - start[axis]) / dir;
        if (t > split_eps && t < 1.0 - split_eps) {
          events.t_values.push_back(t);
          events.wrap_t_values[axis].push_back(t);
        }
      }
    }
  }

  return events;
}

// Appends parameters t in (0, 1), at which the segment crosses a cell face.
// Coordinates are shifted by offset, so that the faces sit at integers n:
//
//      n=2      n=3      n=4      n=5
//   ----|--------|--------|--------|----
//          s---------------->e
//          ^        ^        ^
//        start    t(3)     t(4)       n_begin = floor(start) + 1 = 3
//                                     n_end   = floor(end - eps) = 4
//
// For dir < 0 the faces are walked downwards: ceil(start) - 1 .. ceil(end).
//
void collect_cell_split_params(std::vector<PetscReal>& t_values,
  const DriftKineticSegment& segment, PetscReal offset)
{
  for (Axis axis : {X, Y, Z}) {
    PetscReal start = segment.Rs0[axis] - offset;
    PetscReal end = segment.Rsn[axis] - offset;
    PetscReal dir = end - start;

    if (std::abs(dir) <= split_eps)
      continue;

    if (dir > 0) {
      auto n_begin = (PetscInt)std::floor(start) + 1;
      auto n_end = (PetscInt)std::floor(end - split_eps);

      for (PetscInt n = n_begin; n <= n_end; n++) {
        PetscReal t = ((PetscReal)n - start) / dir;
        if (t > split_eps && t < 1.0 - split_eps)
          t_values.push_back(t);
      }
    }
    else {
      auto n_begin = (PetscInt)std::ceil(start) - 1;
      auto n_end = (PetscInt)std::ceil(end + split_eps);

      for (PetscInt n = n_begin; n >= n_end; n--) {
        PetscReal t = ((PetscReal)n - start) / dir;
        if (t > split_eps && t < 1.0 - split_eps)
          t_values.push_back(t);
      }
    }
  }
}

void clamp_and_check_endpoints(
  Vector3R& start, Vector3R& end, const PeriodicBox (&boxes)[3])
{
  for (Axis axis : {X, Y, Z}) {
    const auto& box = boxes[axis];
    start[axis] = clamp_periodic_coordinate(start[axis], box);
    end[axis] = clamp_periodic_coordinate(end[axis], box);

    PetscReal lo = box.lower - box.tolerance;
    PetscReal hi = box.upper + box.tolerance;

    PetscCheckAbort(start[axis] >= lo && start[axis] <= hi, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
      "Periodic segment start is outside bounds on axis %d: %.16e, [%.16e, %.16e]", axis, start[axis], box.lower, box.upper);

    PetscCheckAbort(end[axis] >= lo && end[axis] <= hi, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
      "Periodic segment end is outside bounds on axis %d: %.16e, [%.16e, %.16e]", axis, end[axis], box.lower, box.upper);
  }
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

std::vector<DriftKineticSegment> periodic_segments(
  const DriftKineticSegment& track, CellSplitMode mode)
{
  Vector3R start = track.Rs0;
  Vector3R end = track.Rsn;
  PeriodicBox boxes[3];

  for (Axis axis : {X, Y, Z}) {
    boxes[axis] = make_periodic_box(axis, mode);
    canonicalize_to_periodic_box(start[axis], end[axis], boxes[axis]);
  }

  auto canonical_track = make_segment(start, end);
  if (canonical_track.dRs_len <= PETSC_SMALL)
    return {canonical_track};

  auto split_events = collect_periodic_split_params(start, end, boxes);
  auto& t_values = split_events.t_values;
  if (t_values.empty()) {
    clamp_and_check_endpoints(start, end, boxes);
    return {make_segment(start, end)};
  }

  t_values.push_back(0);
  t_values.push_back(1);
  sort_unique_t_values(t_values);

  Vector3R dir = canonical_track.dRs;
  Vector3R shift;
  PetscInt wrap_index[3]{0, 0, 0};

  std::vector<DriftKineticSegment> segments;
  segments.reserve(t_values.size() - 1);

  for (PetscInt i = 1; i < (PetscInt)t_values.size(); i++) {
    PetscReal t0 = t_values[i - 1];
    PetscReal t1 = t_values[i - 0];

    Vector3R Rs0 = start + dir * t0 + shift;
    Vector3R Rsn = start + dir * t1 + shift;

    clamp_and_check_endpoints(Rs0, Rsn, boxes);

    auto segment = make_segment(Rs0, Rsn);
    if (segment.dRs_len > split_eps)
      segments.push_back(segment);

    for (Axis axis : {X, Y, Z}) {
      const auto& wrap_t_values = split_events.wrap_t_values[axis];
      while (wrap_index[axis] < (PetscInt)wrap_t_values.size() &&
        std::abs(t1 - wrap_t_values[wrap_index[axis]]) <= split_tie_eps) {
        shift[axis] += split_events.wrap_delta[axis];
        wrap_index[axis]++;
      }
    }
  }

  if (!segments.empty())
    return segments;

  return {canonical_track};
}

std::vector<DriftKineticSegment> cell_segments(
  const std::vector<DriftKineticSegment>& segments, CellSplitMode mode)
{
  PetscReal offset = ((mode == CellSplitMode::centers) ? 0.5 : 0);

  std::vector<DriftKineticSegment> result;
  result.reserve(segments.size());

  for (const auto& segment : segments) {
    if (segment.dRs_len <= PETSC_SMALL) {
      result.push_back(segment);
      continue;
    }

    std::vector<PetscReal> t_values;
    collect_cell_split_params(t_values, segment, offset);
    if (t_values.empty()) {
      result.push_back(segment);
      continue;
    }

    t_values.push_back(0);
    t_values.push_back(1);
    sort_unique_t_values(t_values);

    auto previous_size = result.size();
    for (PetscInt i = 1; i < (PetscInt)(t_values.size()); i++) {
      PetscReal t0 = t_values[i - 1];
      PetscReal t1 = t_values[i - 0];

      Vector3R Rs0 = segment.Rs0 + segment.dRs * t0;
      Vector3R Rsn = segment.Rs0 + segment.dRs * t1;

      auto subsegment = make_segment(Rs0, Rsn);
      if (subsegment.dRs_len > split_eps) {
        result.push_back(subsegment);
      }
    }

    if (result.size() == previous_size)
      result.push_back(segment);
  }

  return result;
}

}  // namespace drift_kinetic
