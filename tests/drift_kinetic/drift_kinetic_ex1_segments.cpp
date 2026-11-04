#include "src/impls/drift_kinetic/segments.h"

namespace {

using namespace drift_kinetic;
using drift_kinetic::CellSplitMode;
using drift_kinetic::DriftKineticSegment;

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
  const std::vector<Endpoints>& expected)
{
  PetscFunctionBeginUser;

  PetscCheck(actual.size() == expected.size(), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
    "%s: expected %zu segments, got %zu", name.c_str(), expected.size(), actual.size());

  Vector3R displacement;
  PetscReal length = 0;
  const Vector3R direction = track.end - track.begin;
  for (std::size_t i = 0; i < actual.size(); i++) {
    const auto& s = actual[i];

    PetscCheck(close(s.Rs0, expected[i].begin) && close(s.Rsn, expected[i].end), PETSC_COMM_WORLD, PETSC_ERR_ARG_CORRUPT,
      "%s: incorrect endpoints of segment %zu", name.c_str(), i);

    PetscCheck(close(s.Rsh, 0.5 * (s.Rs0 + s.Rsn)) && close(s.dRs, s.Rsn - s.Rs0) && abs(s.dRs_len - s.dRs.length()) <= tolerance, PETSC_COMM_WORLD, PETSC_ERR_ARG_CORRUPT,
      "%s: inconsistent segment %zu", name.c_str(), i);

    PetscCheck(actual.size() == 1 || s.dRs_len > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_CORRUPT,
      "%s: empty segment %zu", name.c_str(), i);

    PetscCheck(s.dRs.dot(direction) >= -tolerance && s.dRs.cross(direction).length() <= tolerance * std::max(1.0, direction.length()), PETSC_COMM_WORLD, PETSC_ERR_ARG_CORRUPT,
      "%s: segment %zu changes direction", name.c_str(), i);

    for (Axis axis : {X, Y, Z}) {
      PetscCheck(0 <= s.cell[axis] && s.cell[axis] < Geom_n[axis], PETSC_COMM_WORLD, PETSC_ERR_ARG_CORRUPT,
        "%s: segment %zu outside periodic box", name.c_str(), i);

      const PetscReal node = s.cell[axis];
      PetscCheck(
        s.Rs0[axis] >= node - 0.5 - tolerance &&
        s.Rsn[axis] >= node - 0.5 - tolerance &&
        s.Rs0[axis] <= node + 0.5 + tolerance &&
        s.Rsn[axis] <= node + 0.5 + tolerance,
        PETSC_COMM_WORLD, PETSC_ERR_ARG_CORRUPT,
        "%s: segment %zu crosses a cell plane", name.c_str(), i);

      if (i > 0) {
        PetscReal gap = s.Rs0[axis] - actual[i - 1].Rsn[axis];
        gap -= round(gap / Geom_n[axis]) * Geom_n[axis];

        PetscCheck(abs(gap) <= tolerance, PETSC_COMM_WORLD, PETSC_ERR_ARG_CORRUPT,
          "%s: discontinuity before segment %zu", name.c_str(), i);
      }
    }
    displacement += s.dRs;
    length += s.dRs_len;
  }

  PetscCheck(close(displacement, direction) && abs(length - direction.length()) <= tolerance * std::max(1.0, direction.length()), PETSC_COMM_WORLD, PETSC_ERR_ARG_CORRUPT,
    "%s: displacement or length was lost", name.c_str());

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_case(const std::string& name, const Endpoints& endpoints,
  const std::vector<Endpoints>& expected)
{
  PetscFunctionBeginUser;
  const auto track =
    make_track(physical(endpoints.begin), physical(endpoints.end));

  PetscCheck(
    close(track.Rs0, endpoints.begin) &&
    close(track.Rsn, endpoints.end) &&
    close(make_begin(track), physical(endpoints.begin)) &&
    close(make_end(track), physical(endpoints.end)),
    PETSC_COMM_WORLD, PETSC_ERR_ARG_CORRUPT,
    "%s: coordinate conversion failed", name.c_str());

  const auto actual = cell_segments(track, CellSplitMode::centers);
  PetscCall(check_segments(name, endpoints, actual, expected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The prescribed crossing times define the expected intervals. Periodic
// images are chosen using interval midpoints, independently of the splitter.
std::vector<Endpoints> intervals(
  const Endpoints& track, std::vector<PetscReal> times)
{
  times.push_back(0);
  times.push_back(1);
  std::sort(times.begin(), times.end());
  times.erase(std::unique(times.begin(), times.end()), times.end());
  std::vector<Endpoints> expected;
  for (std::size_t i = 1; i < times.size(); i++) {
    Endpoints s{track.begin + (track.end - track.begin) * times[i - 1],
      track.begin + (track.end - track.begin) * times[i]};
    for (Axis axis : {X, Y, Z}) {
      const PetscReal midpoint = 0.5 * (s.begin[axis] + s.end[axis]);
      const PetscReal shift =
        floor((midpoint + 0.5) / Geom_n[axis]) * Geom_n[axis];
      s.begin[axis] -= shift;
      s.end[axis] -= shift;
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

  // Local segments must retain the original image and cell indices, even
  // when the trajectory crosses either periodic face in either direction.
  for (Axis axis : {X, Y, Z}) {
    for (PetscInt side = 0; side < 2; ++side) {
      for (PetscInt reverse = 0; reverse < 2; ++reverse) {
        Vector3R begin{1, 1, 1}, end = begin;
        const PetscReal face = side ? (PetscReal)Geom_n[axis] - 0.5 : -0.5;
        begin[axis] = face + (reverse ? 0.25 : -0.25);
        end[axis] = face + (reverse ? -0.25 : 0.25);
        const auto track = make_track(physical(begin), physical(end));
        const auto segments = cell_segments(track, CellSplitMode::centers, SegmentImage::local);
        PetscCheck(segments.size() == 2 && close(segments.front().Rs0, begin) &&
            close(segments.back().Rsn, end) && close(segments[0].Rsn, segments[1].Rs0),
          PETSC_COMM_WORLD, PETSC_ERR_USER, "Local seam track is discontinuous");
        Vector3R sum;
        PetscReal length = 0;
        for (const auto& segment : segments) {
          PetscCheck(close(segment.dRs, segment.Rsn - segment.Rs0) &&
              std::abs(segment.dRs_len - segment.dRs.length()) <= tolerance,
            PETSC_COMM_WORLD, PETSC_ERR_USER, "Local segment geometry is inconsistent");
          sum += segment.dRs;
          length += segment.dRs_len;
          for (Axis a : {X, Y, Z})
            PetscCheck(segment.cell[a] == (PetscInt)std::floor(segment.Rsh[a] + 0.5),
              PETSC_COMM_WORLD, PETSC_ERR_USER, "Local cell does not match segment midpoint");
        }
        PetscCheck(close(sum, end - begin) && std::abs(length - (end - begin).length()) <= tolerance,
          PETSC_COMM_WORLD, PETSC_ERR_USER,
          "Local segment displacement changed");
        ++count;
      }
    }
  }

  struct LineCase {
    const char* name;
    PetscReal begin, end;
    std::vector<PetscReal> cuts;
  };
  // An endpoint an ulp past a plane leaves a sliver at that plane.
  const PetscReal above_plane = std::nextafter(0.5, 1.0);
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
    {"almost_ends_on_plane", 0.25, above_plane, {0.25 / (above_plane - 0.25)}},
  };
  for (Axis axis : {X, Y, Z}) {
    for (const auto& c : line_cases) {
      for (PetscInt reverse = 0; reverse < 2; reverse++) {
        Endpoints track{};
        track.begin[axis] = reverse ? c.end : c.begin;
        track.end[axis] = reverse ? c.begin : c.end;
        auto cuts = c.cuts;
        if (reverse)
          for (auto& t : cuts)
            t = 1 - t;
        PetscCall(check_case(std::format("cell/{}/{}/{}", c.name, (PetscInt)axis, reverse), track, intervals(track, cuts)));
        count++;
      }
    }

    const PetscReal length = Geom_n[axis];
    const PetscReal upper = length - 0.5;
    const PetscReal below_upper = std::nextafter(upper, PetscReal{0});
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
      {"almost_upper", below_upper, upper + 0.25, {(upper - below_upper) / (upper + 0.25 - below_upper)}},
    };
    for (const auto& c : periodic_cases) {
      for (PetscInt reverse = 0; reverse < 2; reverse++) {
        Endpoints track{};
        track.begin[axis] = reverse ? c.end : c.begin;
        track.end[axis] = reverse ? c.begin : c.end;
        auto cuts = c.cuts;
        if (reverse)
          for (auto& t : cuts)
            t = 1 - t;
        // These are short tracks or whole boxes with integer cell counts.
        // Enumerate the known half-integer planes for the expected cell cuts.
        auto cell_cuts = cuts;
        for (PetscInt plane = -1; plane <= 2 * Geom_n[axis]; plane++) {
          const PetscReal position = plane + 0.5;
          if (position > std::min(track.begin[axis], track.end[axis]) + 1e-12 &&
            position < std::max(track.begin[axis], track.end[axis]) - 1e-12)
            cell_cuts.push_back((position - track.begin[axis]) /
              (track.end[axis] - track.begin[axis]));
        }
        PetscCall(check_case(std::format("periodic/{}/{}/{}", c.name, (PetscInt)axis, reverse), track, intervals(track, cell_cuts)));
        count++;
      }
    }
  }

  // All 26 nonzero signed directions: six faces, twelve edges, eight corners.
  for (PetscInt sx = -1; sx <= 1; sx++) {
    for (PetscInt sy = -1; sy <= 1; sy++) {
      for (PetscInt sz = -1; sz <= 1; sz++) {
        const Vector3I signs{sx, sy, sz};
        if (signs.abs_max() == 0)
          continue;
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
        PetscCall(check_case(name + "/cell", cell, intervals(cell, {0.5})));
        PetscCall(check_case(name + "/periodic", periodic, intervals(periodic, {0.5})));
        count += 2;
      }
    }
  }

  // Every ordering and tie of three crossing events in all eight octants.
  for (PetscInt octant = 0; octant < 8; octant++) {
    for (PetscInt ordering = 0; ordering < 27; ordering++) {
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
      PetscCall(check_case(name + "/cell", cell, intervals(cell, times)));
      const auto expected = intervals(periodic, times);
      PetscCall(check_case(name + "/periodic", periodic, expected));
      // Moving the complete track by whole periods must preserve the result.
      for (Axis axis : {X, Y, Z}) {
        const PetscReal shift = (axis - 1) * 2 * Geom_n[axis];
        periodic.begin[axis] += shift;
        periodic.end[axis] += shift;
      }
      PetscCall(check_case(name + "/translated", periodic, expected));
      count += 3;
    }
  }

  // Independently place each endpoint before, on, or after a boundary.
  // The Cartesian product covers stationary axes, tangency and mixed events.
  for (PetscInt mask = 0; mask < 8; mask++) {
    for (PetscInt placement = 0; placement < 729; placement++) {
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
      const std::vector<PetscReal> times = crosses ? std::vector<PetscReal>{0.5}
                                                   : std::vector<PetscReal>{};
      const auto name = std::format("placement/{}/{}", mask, placement);
      if (mask == 0) {
        PetscCall(check_case(name + "/cell", cell, intervals(cell, times)));
        count++;
      }
      PetscCall(check_case(name + "/periodic", periodic, intervals(periodic, times)));
      count++;
    }
  }

  // Intersections separated by roundoff are resolved as distinct events.
  for (PetscReal gap : {PetscReal{2e-14}, PetscReal{1e-8}}) {
    const Endpoints cell{{0, -gap, 0}, {1, 1 - gap, 0}};
    const Endpoints periodic{{3, 5 - gap, 0}, {4, 6 - gap, 0}};
    const std::vector<PetscReal> times{0.5, 0.5 + gap};
    const auto name = std::format("near_tie/{}", gap);
    PetscCall(check_case(name + "/cell", cell, intervals(cell, times)));
    PetscCall(check_case(name + "/periodic", periodic, intervals(periodic, times)));
    count += 2;
  }

  // Wrapping a coordinate an ulp below the lower face rounds it onto the
  // upper face, the stored cell must stay inside the box.
  const PetscReal below_lower = std::nextafter(-0.5, PetscReal{-1});
  const Endpoints below{{below_lower, 0, 0}, {below_lower, 0.25, 0}};
  PetscCall(check_case("below_lower", below, intervals(below, {})));
  count++;

  const Endpoints multi{{-0.5, -0.5, -0.5}, {7.5, 11.5, 15.5}};
  std::vector<PetscReal> multi_cell_times;
  for (Axis axis : {X, Y, Z})
    for (PetscInt n = 1; n < 2 * Geom_n[axis]; n++)
      multi_cell_times.push_back(static_cast<PetscReal>(n) / (2 * Geom_n[axis]));
  PetscCall(check_case("multiple_wraps_xyz", multi, intervals(multi, multi_cell_times)));
  count++;

  // Preserve the other supported mode.
  const auto edge_track =
    make_track(physical({0.25, 0, 0}), physical({1.25, 0, 0}));
  const auto edges = cell_segments(edge_track, CellSplitMode::edges);

  PetscCheck(
    edges.size() == 2 &&
    close(edges[0].Rsn, {1, 0, 0}) &&
    close(edges[1].Rs0, {1, 0, 0}) &&
    edges[0].cell == Vector3I(0, 0, 0) &&
    edges[1].cell == Vector3I(1, 0, 0),
    PETSC_COMM_WORLD, PETSC_ERR_ARG_CORRUPT,
    "edges was changed");

  const auto edge_wrap = cell_segments(
    make_track(physical({3.75, 0.25, 0.25}), physical({4.25, 0.25, 0.25})),
    CellSplitMode::edges);

  PetscCheck(
    edge_wrap.size() == 2 &&
    close(edge_wrap[0].Rsn, {4, 0.25, 0.25}) &&
    close(edge_wrap[1].Rs0, {0, 0.25, 0.25}) &&
    edge_wrap[0].cell == Vector3I(3, 0, 0) &&
    edge_wrap[1].cell == Vector3I(0, 0, 0),
    PETSC_COMM_WORLD, PETSC_ERR_ARG_CORRUPT,
    "edges periodic box was changed");

  count += 2;

  Geom_n[X] = Geom_n[Y] = Geom_n[Z] = 1;
  const Endpoints unit{{-0.5, -0.5, -0.5}, {2.5, -3.5, 1.5}};
  PetscCall(check_case("unit_period", unit, intervals(unit, {1.0 / 3, 0.5, 2.0 / 3})));
  count++;
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
