#include "src/impls/drift_kinetic/segments.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <string>
#include <vector>

namespace {

using drift_kinetic::CellSplitMode;
using drift_kinetic::DriftKineticSegment;
constexpr auto centers = CellSplitMode::cell_centers;
constexpr PetscReal tolerance = 2e-12;

struct Endpoints {
  Vector3R begin;
  Vector3R end;
};

Vector3R physical(const Vector3R& r)
{
  return {r[X] * dx, r[Y] * dy, r[Z] * dz};
}

bool close(const Vector3R& a, const Vector3R& b)
{
  return (a - b).abs_max() <= tolerance;
}

PetscErrorCode check_segments(const std::string& name, const Endpoints& track,
  const std::vector<DriftKineticSegment>& actual,
  const std::vector<Endpoints>& expected, bool periodic, bool cells)
{
  PetscFunctionBeginUser;
  PetscCheck(actual.size() == expected.size(), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "%s: expected %zu segments, got %zu", name.c_str(), expected.size(), actual.size());

  Vector3R displacement;
  PetscReal length = 0;
  const Vector3R direction = track.end - track.begin;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto& s = actual[i];
    PetscCheck(close(s.Rs0, expected[i].begin) && close(s.Rsn, expected[i].end),
      PETSC_COMM_WORLD, PETSC_ERR_USER, "%s: incorrect endpoints of segment %zu",
      name.c_str(), i);
    PetscCheck(close(s.Rsmid, 0.5 * (s.Rs0 + s.Rsn)) && close(s.dRs, s.Rsn - s.Rs0) &&
      std::abs(s.dRs_len - s.dRs.length()) <= tolerance,
      PETSC_COMM_WORLD, PETSC_ERR_USER, "%s: inconsistent segment %zu", name.c_str(), i);
    PetscCheck(actual.size() == 1 || s.dRs_len > 0, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "%s: empty segment %zu", name.c_str(), i);
    PetscCheck(s.dRs.dot(direction) >= -tolerance &&
      s.dRs.cross(direction).length() <= tolerance * std::max(PetscReal{1}, direction.length()),
      PETSC_COMM_WORLD, PETSC_ERR_USER, "%s: segment %zu changes direction", name.c_str(), i);

    for (Axis axis : {X, Y, Z}) {
      if (periodic) {
        PetscCheck(s.Rs0[axis] >= -0.5 - tolerance &&
          s.Rsn[axis] >= -0.5 - tolerance &&
          s.Rs0[axis] <= Geom_n[axis] - 0.5 + tolerance &&
          s.Rsn[axis] <= Geom_n[axis] - 0.5 + tolerance,
          PETSC_COMM_WORLD, PETSC_ERR_USER, "%s: segment %zu outside periodic box",
          name.c_str(), i);
      }
      if (cells) {
        const PetscReal node = std::floor(s.Rsmid[axis] + 0.5);
        PetscCheck(s.Rs0[axis] >= node - 0.5 - tolerance &&
          s.Rsn[axis] >= node - 0.5 - tolerance &&
          s.Rs0[axis] <= node + 0.5 + tolerance &&
          s.Rsn[axis] <= node + 0.5 + tolerance,
          PETSC_COMM_WORLD, PETSC_ERR_USER, "%s: segment %zu crosses a cell plane",
          name.c_str(), i);
      }
      if (i > 0) {
        PetscReal gap = s.Rs0[axis] - actual[i - 1].Rsn[axis];
        if (periodic)
          gap -= std::round(gap / Geom_n[axis]) * Geom_n[axis];
        PetscCheck(std::abs(gap) <= tolerance, PETSC_COMM_WORLD, PETSC_ERR_USER,
          "%s: discontinuity before segment %zu", name.c_str(), i);
      }
    }
    displacement += s.dRs;
    length += s.dRs_len;
  }
  PetscCheck(close(displacement, direction) &&
    std::abs(length - direction.length()) <= tolerance * std::max(PetscReal{1}, direction.length()),
    PETSC_COMM_WORLD, PETSC_ERR_USER, "%s: displacement or length was lost", name.c_str());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_cell_case(const std::string& name, const Endpoints& endpoints,
  const std::vector<Endpoints>& expected)
{
  PetscFunctionBeginUser;
  const auto track = drift_kinetic::make_track(physical(endpoints.begin), physical(endpoints.end));
  PetscCheck(close(track.Rs0, endpoints.begin) && close(track.Rsn, endpoints.end) &&
    close(drift_kinetic::make_begin(track), physical(endpoints.begin)) &&
    close(drift_kinetic::make_end(track), physical(endpoints.end)),
    PETSC_COMM_WORLD, PETSC_ERR_USER, "%s: coordinate conversion failed", name.c_str());
  const auto actual = drift_kinetic::cell_segments({track}, centers);
  PetscCall(check_segments(name, endpoints, actual, expected, false, true));
  PetscCall(check_segments(name + "/repeat", endpoints,
    drift_kinetic::cell_segments(actual, centers), expected, false, true));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_periodic_case(const std::string& name, const Endpoints& endpoints,
  const std::vector<Endpoints>& expected, const std::vector<Endpoints>& cell_expected)
{
  PetscFunctionBeginUser;
  const auto track = drift_kinetic::make_track(physical(endpoints.begin), physical(endpoints.end));
  const auto periodic = drift_kinetic::periodic_segments(track, centers);
  PetscCall(check_segments(name + "/periodic", endpoints, periodic, expected, true, false));
  const auto cells = drift_kinetic::cell_segments(periodic, centers);
  PetscCall(check_segments(name + "/cells", endpoints, cells, cell_expected, true, true));
  PetscCall(check_segments(name + "/repeat", endpoints,
    drift_kinetic::cell_segments(cells, centers), cell_expected, true, true));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The prescribed crossing times define the expected intervals. Periodic
// images are chosen using interval midpoints, independently of the splitter.
std::vector<Endpoints> intervals(const Endpoints& track, std::vector<PetscReal> times,
  bool periodic)
{
  times.push_back(0);
  times.push_back(1);
  std::sort(times.begin(), times.end());
  times.erase(std::unique(times.begin(), times.end()), times.end());
  std::vector<Endpoints> expected;
  for (std::size_t i = 1; i < times.size(); ++i) {
    Endpoints s{track.begin + (track.end - track.begin) * times[i - 1],
      track.begin + (track.end - track.begin) * times[i]};
    if (periodic) {
      for (Axis axis : {X, Y, Z}) {
        const PetscReal midpoint = 0.5 * (s.begin[axis] + s.end[axis]);
        const PetscReal shift = std::floor((midpoint + 0.5) / Geom_n[axis]) * Geom_n[axis];
        s.begin[axis] -= shift;
        s.end[axis] -= shift;
      }
    }
    expected.push_back(s);
  }
  return expected;
}

PetscErrorCode run_cases(std::size_t& count)
{
  PetscFunctionBeginUser;
  dx = 0.25;
  dy = 1.5;
  dz = 2;
  Geom_n[X] = 4;
  Geom_n[Y] = 6;
  Geom_n[Z] = 8;

  struct LineCase {
    const char* name;
    PetscReal begin, end;
    std::vector<PetscReal> cuts;
  };
  const std::vector<LineCase> line_cases{
    {"inside", 0, 0.25, {}},
    {"stationary", 0, 0, {}},
    {"stationary_on_plane", 0.5, 0.5, {}},
    {"cross_plane", 0.25, 0.75, {0.5}},
    {"start_on_plane", 0.5, 0.75, {}},
    {"end_on_plane", 0.25, 0.5, {}},
    {"between_planes", -0.5, 0.5, {}},
    {"multiple_cells", -1, 2, {1.0 / 6, 0.5, 5.0 / 6}},
    {"multiple_planes", -1.5, 2.5, {0.25, 0.5, 0.75}},
    {"near_stationary", 0, 1e-11, {}},
    {"almost_on_plane", std::nextafter(0.5, PetscReal{0}), 0.75, {}},
    {"almost_ends_on_plane", 0.25, std::nextafter(0.5, PetscReal{1}), {}},
  };
  for (Axis axis : {X, Y, Z}) {
    for (const auto& c : line_cases) {
      for (PetscInt reverse = 0; reverse < 2; ++reverse) {
        Endpoints track{};
        track.begin[axis] = reverse ? c.end : c.begin;
        track.end[axis] = reverse ? c.begin : c.end;
        auto cuts = c.cuts;
        if (reverse)
          for (auto& t : cuts) t = 1 - t;
        PetscCall(check_cell_case(std::format("cell/{}/{}/{}", c.name, static_cast<PetscInt>(axis), reverse),
          track, intervals(track, cuts, false)));
        ++count;
      }
    }

    const PetscReal length = Geom_n[axis];
    const PetscReal upper = length - 0.5;
    const std::vector<LineCase> periodic_cases{
      {"inside", 0, 0.25, {}},
      {"stationary", 0, 0, {}},
      {"stationary_lower", -0.5, -0.5, {}},
      {"stationary_upper", upper, upper, {}},
      {"cross_upper", upper - 0.25, upper + 0.25, {0.5}},
      {"cross_lower", -0.25, -0.75, {0.5}},
      {"start_lower_inward", -0.5, -0.25, {}},
      {"start_lower_outward", -0.5, -0.75, {}},
      {"start_upper_inward", upper, upper - 0.25, {}},
      {"start_upper_outward", upper, upper + 0.25, {}},
      {"whole_box", -0.5, upper, {}},
      {"multiple_wraps", -0.5, upper + length, {0.5}},
      {"near_stationary", 0, 1e-11, {}},
      {"almost_lower", std::nextafter(-0.5, PetscReal{-1}), -0.75, {}},
      {"almost_upper", std::nextafter(upper, PetscReal{0}), upper + 0.25, {}},
    };
    for (const auto& c : periodic_cases) {
      for (PetscInt reverse = 0; reverse < 2; ++reverse) {
        Endpoints track{};
        track.begin[axis] = reverse ? c.end : c.begin;
        track.end[axis] = reverse ? c.begin : c.end;
        auto cuts = c.cuts;
        if (reverse)
          for (auto& t : cuts) t = 1 - t;
        // These are short tracks or whole boxes with integer cell counts.
        // Enumerate the known half-integer planes for the expected cell cuts.
        auto cell_cuts = cuts;
        for (PetscInt plane = -1; plane <= 2 * Geom_n[axis]; ++plane) {
          const PetscReal position = plane + 0.5;
          if (position > std::min(track.begin[axis], track.end[axis]) + 1e-12 &&
            position < std::max(track.begin[axis], track.end[axis]) - 1e-12)
            cell_cuts.push_back((position - track.begin[axis]) / (track.end[axis] - track.begin[axis]));
        }
        PetscCall(check_periodic_case(std::format("periodic/{}/{}/{}", c.name, static_cast<PetscInt>(axis), reverse),
          track, intervals(track, cuts, true), intervals(track, cell_cuts, true)));
        ++count;
      }
    }
  }

  // All 26 nonzero signed directions: six faces, twelve edges, eight corners.
  for (PetscInt sx = -1; sx <= 1; ++sx) {
    for (PetscInt sy = -1; sy <= 1; ++sy) {
      for (PetscInt sz = -1; sz <= 1; ++sz) {
        const Vector3I signs{sx, sy, sz};
        if (signs.abs_max() == 0) continue;
        Endpoints cell{}, periodic{};
        for (Axis axis : {X, Y, Z}) {
          const PetscReal sign = signs[axis];
          cell.begin[axis] = sign * 0.25;
          cell.end[axis] = sign * 0.75;
          const PetscReal boundary = sign > 0 ? Geom_n[axis] - 0.5 : -0.5;
          periodic.begin[axis] = sign == 0 ? 0 : boundary - sign * 0.25;
          periodic.end[axis] = sign == 0 ? 0 : boundary + sign * 0.25;
        }
        const auto name = std::format("direction/{}/{}/{}", sx, sy, sz);
        PetscCall(check_cell_case(name, cell, intervals(cell, {0.5}, false)));
        const auto expected = intervals(periodic, {0.5}, true);
        PetscCall(check_periodic_case(name, periodic, expected, expected));
        count += 2;
      }
    }
  }

  // Every ordering and tie of three crossing events in all eight octants.
  for (PetscInt octant = 0; octant < 8; ++octant) {
    for (PetscInt ordering = 0; ordering < 27; ++ordering) {
      Endpoints cell{}, periodic{};
      std::vector<PetscReal> times;
      PetscInt code = ordering;
      for (Axis axis : {X, Y, Z}) {
        const PetscReal t = 0.25 * (1 + code % 3);
        code /= 3;
        times.push_back(t);
        const PetscReal sign = octant & (1 << axis) ? 1 : -1;
        cell.begin[axis] = sign * (0.5 - t);
        cell.end[axis] = cell.begin[axis] + sign;
        const PetscReal boundary = sign > 0 ? Geom_n[axis] - 0.5 : -0.5;
        periodic.begin[axis] = boundary - sign * t;
        periodic.end[axis] = periodic.begin[axis] + sign;
      }
      const auto name = std::format("ordering/{}/{}", octant, ordering);
      PetscCall(check_cell_case(name, cell, intervals(cell, times, false)));
      const auto expected = intervals(periodic, times, true);
      PetscCall(check_periodic_case(name, periodic, expected, expected));
      // Moving the complete track by whole periods must preserve the result.
      for (Axis axis : {X, Y, Z}) {
        const PetscReal shift = (axis - 1) * 2 * Geom_n[axis];
        periodic.begin[axis] += shift;
        periodic.end[axis] += shift;
      }
      PetscCall(check_periodic_case(name + "/translated", periodic, expected, expected));
      count += 3;
    }
  }

  // Independently place each endpoint before, on, or after a boundary.
  // The Cartesian product covers stationary axes, tangency and mixed events.
  for (PetscInt mask = 0; mask < 8; ++mask) {
    for (PetscInt placement = 0; placement < 729; ++placement) {
      Endpoints cell{}, periodic{};
      bool crosses = false;
      PetscInt code = placement;
      for (Axis axis : {X, Y, Z}) {
        const PetscReal begin = 0.25 * (code % 3 - 1);
        code /= 3;
        const PetscReal end = 0.25 * (code % 3 - 1);
        code /= 3;
        crosses = crosses || begin * end < 0;
        cell.begin[axis] = 0.5 + begin;
        cell.end[axis] = 0.5 + end;
        const PetscReal boundary = mask & (1 << axis) ? Geom_n[axis] - 0.5 : -0.5;
        periodic.begin[axis] = boundary + begin;
        periodic.end[axis] = boundary + end;
      }
      const std::vector<PetscReal> times = crosses ? std::vector<PetscReal>{0.5} : std::vector<PetscReal>{};
      const auto name = std::format("placement/{}/{}", mask, placement);
      if (mask == 0) {
        PetscCall(check_cell_case(name, cell, intervals(cell, times, false)));
        ++count;
      }
      const auto expected = intervals(periodic, times, true);
      PetscCall(check_periodic_case(name, periodic, expected, expected));
      ++count;
    }
  }

  PetscCheck(drift_kinetic::cell_segments({}, centers).empty(),
    PETSC_COMM_WORLD, PETSC_ERR_USER, "Empty input must stay empty");
  ++count;

  // Resolve distinct events but merge intersections separated by roundoff.
  for (PetscReal gap : {PetscReal{2e-14}, PetscReal{1e-8}}) {
    const Endpoints cell{{0, -gap, 0}, {1, 1 - gap, 0}};
    const Endpoints periodic{{3, 5 - gap, 0}, {4, 6 - gap, 0}};
    std::vector<PetscReal> times{0.5};
    if (gap > 1e-12) times.push_back(0.5 + gap);
    const auto name = std::format("near_tie/{}", gap);
    PetscCall(check_cell_case(name, cell, intervals(cell, times, false)));
    const auto expected = intervals(periodic, times, true);
    PetscCall(check_periodic_case(name, periodic, expected, expected));
    count += 2;
  }

  const Endpoints multi{{-0.5, -0.5, -0.5}, {7.5, 11.5, 15.5}};
  std::vector<PetscReal> multi_cell_times;
  for (Axis axis : {X, Y, Z})
    for (PetscInt n = 1; n < 2 * Geom_n[axis]; ++n)
      multi_cell_times.push_back(static_cast<PetscReal>(n) / (2 * Geom_n[axis]));
  PetscCall(check_periodic_case("multiple_wraps_xyz", multi,
    intervals(multi, {0.5}, true), intervals(multi, multi_cell_times, true)));
  ++count;

  const Endpoints first{{0, 0, 0}, {0.75, 0, 0}};
  const Endpoints second{{2.75, 0, 0}, {2.25, 0, 0}};
  const auto batch = drift_kinetic::cell_segments({
    drift_kinetic::make_track(physical(first.begin), physical(first.end)),
    drift_kinetic::make_track(physical(second.begin), physical(second.end))}, centers);
  PetscCheck(batch.size() == 4, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Batch processing lost input segments");
  PetscCall(check_segments("batch/first", first, {batch.begin(), batch.begin() + 2},
    intervals(first, {2.0 / 3}, false), false, true));
  PetscCall(check_segments("batch/second", second, {batch.begin() + 2, batch.end()},
    intervals(second, {0.5}, false), false, true));
  ++count;

  // Preserve the other supported mode and its default argument.
  const auto edge_track = drift_kinetic::make_track(physical({0.25, 0, 0}), physical({1.25, 0, 0}));
  const auto edges = drift_kinetic::cell_segments({edge_track});
  PetscCheck(edges.size() == 2 && close(edges[0].Rsn, {1, 0, 0}) &&
    close(edges[1].Rs0, {1, 0, 0}), PETSC_COMM_WORLD, PETSC_ERR_USER, "cell_edges was changed");
  const auto edge_wrap = drift_kinetic::periodic_segments(
    drift_kinetic::make_track(physical({3.75, 0.25, 0.25}), physical({4.25, 0.25, 0.25})));
  PetscCheck(edge_wrap.size() == 2 && close(edge_wrap[0].Rsn, {4, 0.25, 0.25}) &&
    close(edge_wrap[1].Rs0, {0, 0.25, 0.25}),
    PETSC_COMM_WORLD, PETSC_ERR_USER, "cell_edges periodic box was changed");
  count += 2;

  Geom_n[X] = Geom_n[Y] = Geom_n[Z] = 1;
  const Endpoints unit{{-0.5, -0.5, -0.5}, {2.5, -3.5, 1.5}};
  const auto unit_expected = intervals(unit, {1.0 / 3, 0.5, 2.0 / 3}, true);
  PetscCall(check_periodic_case("unit_period", unit, unit_expected, unit_expected));
  ++count;
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, "Drift-kinetic trajectory segmentation tests.\n"));
  std::size_t count = 0;
  PetscCall(run_cases(count));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Checked %zu segmentation cases.\n", count));
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
