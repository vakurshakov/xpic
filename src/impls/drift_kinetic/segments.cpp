#include "segments.h"

namespace drift_kinetic {

constexpr PetscReal split_eps = 1e-14;
constexpr PetscReal split_tie_eps = 1e-13;

struct PeriodicBox {
  PetscReal lower = 0.0;
  PetscReal upper = 0.0;
  PetscReal length = 0.0;
};

DriftKineticSegment make_segment(const Vector3R& Rs0, const Vector3R& Rsn)
{
  DriftKineticSegment segment;
  segment.Rs0 = Rs0;
  segment.Rsn = Rsn;
  segment.Rsmid = 0.5 * (Rs0 + Rsn);
  segment.dRs = Rsn - Rs0;
  segment.dRs_len = segment.dRs.length();
  return segment;
}

PeriodicBox make_periodic_box(Axis axis, CellSplitMode mode)
{
  const PetscReal length = static_cast<PetscReal>(Geom_n[axis]);
  PetscCheckAbort(length > split_eps, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
    "Invalid periodic length on axis %d: Geom_n[%d]=%.16e",
    axis, axis, length);

  if (mode == CellSplitMode::cell_centers)
    //return PeriodicBox{0.5, length + 0.5, length};
    return PeriodicBox{-0.5, length - 0.5, length};
  return PeriodicBox{0.0, length, length};
}

PetscReal clamp_periodic_coordinate(PetscReal value, const PeriodicBox& box)
{
  if (std::abs(value - box.lower) <= split_eps)
    return box.lower;
  if (std::abs(value - box.upper) <= split_eps)
    return box.upper;
  if (value < box.lower && value > box.lower - split_eps)
    return box.lower;
  if (value > box.upper && value < box.upper + split_eps)
    return box.upper;
  return value;
}

void sort_unique_t_values(std::vector<PetscReal>& t_values)
{
  std::sort(t_values.begin(), t_values.end());

  std::vector<PetscReal> unique;
  unique.reserve(t_values.size());

  for (PetscReal t : t_values) {
    t = std::clamp(t, 0.0, 1.0);
    if (unique.empty() || std::abs(t - unique.back()) > split_tie_eps)
      unique.push_back(t);
  }

  t_values.swap(unique);
}

void canonicalize_to_periodic_box(PetscReal& start, PetscReal& end, const PeriodicBox& box)
{
  const PetscReal image_shift = std::floor((start - box.lower) / box.length);
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

  const PetscReal dir = end - start;
  if (dir < -split_eps && std::abs(start - box.lower) <= split_eps) {
    start += box.length;
    end += box.length;
  }
  else if (dir > split_eps && std::abs(start - box.upper) <= split_eps) {
    start -= box.length;
    end -= box.length;
  }

  start = clamp_periodic_coordinate(start, box);
}

struct PeriodicSplitEvents {
  std::vector<PetscReal> t_values;
  std::array<std::vector<PetscReal>, 3> wrap_t_values;
  std::array<PetscReal, 3> wrap_delta{0.0, 0.0, 0.0};
};

PeriodicSplitEvents collect_periodic_split_params(
  const Vector3R& start,
  const Vector3R& end,
  CellSplitMode mode)
{
  PeriodicSplitEvents events;

  for (Axis axis : {X, Y, Z}) {
    const PeriodicBox box = make_periodic_box(axis, mode);
    const PetscReal dir = end[axis] - start[axis];
    const PetscReal start_shifted = start[axis] - box.lower;
    const PetscReal end_shifted = end[axis] - box.lower;

    if (std::abs(dir) <= split_eps)
      continue;

    if (dir > 0.0) {
      events.wrap_delta[axis] = -box.length;

      const PetscInt k_begin =
        static_cast<PetscInt>(std::floor(start_shifted / box.length)) + 1;
      const PetscInt k_end =
        static_cast<PetscInt>(std::floor((end_shifted - split_eps) / box.length));

      for (PetscInt k = k_begin; k <= k_end; ++k) {
        const PetscReal boundary = box.lower + static_cast<PetscReal>(k) * box.length;
        const PetscReal t = (boundary - start[axis]) / dir;
        if (t > split_eps && t < 1.0 - split_eps) {
          events.t_values.push_back(t);
          events.wrap_t_values[axis].push_back(t);
        }
      }
    }
    else {
      events.wrap_delta[axis] = box.length;

      const PetscInt k_begin =
        static_cast<PetscInt>(std::ceil(start_shifted / box.length)) - 1;
      const PetscInt k_end =
        static_cast<PetscInt>(std::ceil((end_shifted + split_eps) / box.length));

      for (PetscInt k = k_begin; k >= k_end; --k) {
        const PetscReal boundary = box.lower + static_cast<PetscReal>(k) * box.length;
        const PetscReal t = (boundary - start[axis]) / dir;
        if (t > split_eps && t < 1.0 - split_eps) {
          events.t_values.push_back(t);
          events.wrap_t_values[axis].push_back(t);
        }
      }
    }
  }

  return events;
}

void collect_cell_split_params(std::vector<PetscReal>& t_values,
  const DriftKineticSegment& segment, PetscReal offset = 0.0)
{
  for (Axis axis : {X, Y, Z}) {
    const PetscReal start = segment.Rs0[axis] - offset;
    const PetscReal end = segment.Rsn[axis] - offset;
    const PetscReal dir = end - start;

    if (std::abs(dir) <= split_eps)
      continue;

    if (dir > 0.0) {
      const PetscInt n_begin = static_cast<PetscInt>(std::floor(start)) + 1;
      const PetscInt n_end = static_cast<PetscInt>(std::floor(end - split_eps));

      for (PetscInt n = n_begin; n <= n_end; ++n) {
        const PetscReal t = (static_cast<PetscReal>(n) - start) / dir;
        if (t > split_eps && t < 1.0 - split_eps)
          t_values.push_back(t);
      }
    }
    else {
      const PetscInt n_begin = static_cast<PetscInt>(std::ceil(start)) - 1;
      const PetscInt n_end = static_cast<PetscInt>(std::ceil(end + split_eps));

      for (PetscInt n = n_begin; n >= n_end; --n) {
        const PetscReal t = (static_cast<PetscReal>(n) - start) / dir;
        if (t > split_eps && t < 1.0 - split_eps)
          t_values.push_back(t);
      }
    }
  }
}

DriftKineticSegment make_track(const Vector3R& R0, const Vector3R& Rn) {
  return make_segment(::Shape::make_r(R0), ::Shape::make_r(Rn));
}

Vector3R make_mid(const DriftKineticSegment& segment) {
    return {segment.Rsmid.x() * dx,
            segment.Rsmid.y() * dy,
            segment.Rsmid.z() * dz};
}

Vector3R make_end(const DriftKineticSegment& segment) {
    return {segment.Rsn.x() * dx,
            segment.Rsn.y() * dy,
            segment.Rsn.z() * dz};
}

Vector3R make_begin(const DriftKineticSegment& segment) {
    return {segment.Rs0.x() * dx,
            segment.Rs0.y() * dy,
            segment.Rs0.z() * dz};
}

std::vector<DriftKineticSegment> periodic_segments(
  const DriftKineticSegment& track,
  CellSplitMode mode)
{
  Vector3R start = track.Rs0;
  Vector3R end = track.Rsn;

  for (Axis axis : {X, Y, Z}) {
    canonicalize_to_periodic_box(start[axis], end[axis], make_periodic_box(axis, mode));
  }

  const DriftKineticSegment canonical_track = make_segment(start, end);
  if (canonical_track.dRs_len <= PETSC_SMALL)
    return {canonical_track};

  const PeriodicSplitEvents split_events = collect_periodic_split_params(start, end, mode);

  std::vector<PetscReal> t_values;
  t_values.reserve(split_events.t_values.size() + 2);
  t_values.push_back(0.0);
  t_values.insert(t_values.end(), split_events.t_values.begin(), split_events.t_values.end());
  t_values.push_back(1.0);
  sort_unique_t_values(t_values);

  const Vector3R dir = end - start;
  std::array<PetscInt, 3> wrap_index{0, 0, 0};
  Vector3R shift{};
  std::vector<DriftKineticSegment> segments;
  segments.reserve(t_values.size());

  for (PetscInt i = 1; i < static_cast<PetscInt>(t_values.size()); ++i) {
    const PetscReal t0 = t_values[i - 1];
    const PetscReal t1 = t_values[i];

    Vector3R Rs0 = start + dir * t0 + shift;
    Vector3R Rsn = start + dir * t1 + shift;

    for (Axis axis : {X, Y, Z}) {
      const PeriodicBox box = make_periodic_box(axis, mode);
      Rs0[axis] = clamp_periodic_coordinate(Rs0[axis], box);
      Rsn[axis] = clamp_periodic_coordinate(Rsn[axis], box);

      PetscCheckAbort(Rs0[axis] >= box.lower - split_eps && Rs0[axis] <= box.upper + split_eps,
        PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
        "Periodic segment start is outside bounds on axis %d: %.16e, [%.16e, %.16e]",
        axis, Rs0[axis], box.lower, box.upper);
      PetscCheckAbort(Rsn[axis] >= box.lower - split_eps && Rsn[axis] <= box.upper + split_eps,
        PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
        "Periodic segment end is outside bounds on axis %d: %.16e, [%.16e, %.16e]",
        axis, Rsn[axis], box.lower, box.upper);
    }

    const DriftKineticSegment segment = make_segment(Rs0, Rsn);
    if (segment.dRs_len > split_eps)
      segments.push_back(segment);

    for (Axis axis : {X, Y, Z}) {
      const auto& wrap_t_values = split_events.wrap_t_values[axis];
      while (wrap_index[axis] < static_cast<PetscInt>(wrap_t_values.size()) &&
        std::abs(t1 - wrap_t_values[wrap_index[axis]]) <= split_tie_eps) {
        shift[axis] += split_events.wrap_delta[axis];
        ++wrap_index[axis];
      }
    }
  }

  if (!segments.empty())
    return segments;
  return {canonical_track};
}

std::vector<DriftKineticSegment> cell_segments(
  const std::vector<DriftKineticSegment>& segments,
  CellSplitMode mode)
{
  const PetscReal offset = (mode == CellSplitMode::cell_centers) ? 0.5 : 0.0;

  std::vector<DriftKineticSegment> result;
  result.reserve(segments.size());

  for (const auto& input_segment : segments) {
    const DriftKineticSegment segment = make_segment(input_segment.Rs0, input_segment.Rsn);
    if (segment.dRs_len <= PETSC_SMALL) {
      result.push_back(segment);
      continue;
    }

    std::vector<PetscReal> t_values{0.0, 1.0};
    collect_cell_split_params(t_values, segment, offset);
    sort_unique_t_values(t_values);

    bool added = false;
    for (PetscInt i = 1; i < static_cast<PetscInt>(t_values.size()); ++i) {
      const PetscReal t0 = t_values[i - 1];
      const PetscReal t1 = t_values[i];

      const DriftKineticSegment subsegment = make_segment(
        segment.Rs0 + segment.dRs * t0,
        segment.Rs0 + segment.dRs * t1);
      if (subsegment.dRs_len > split_eps) {
        result.push_back(subsegment);
        added = true;
      }
    }

    if (!added)
      result.push_back(segment);
  }

  return result;
}
}
