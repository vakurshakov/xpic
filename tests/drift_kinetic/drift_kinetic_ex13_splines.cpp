#include "src/algorithms/implicit_drift_kinetic.h"

namespace {

using namespace drift_kinetic;

constexpr char help[] =
  "Compare drift-kinetic form factors with independent spline integrals.\n";

PetscReal linear(PetscReal r)
{
  return std::max(PetscReal{0}, 1 - std::abs(r));
}

PetscReal quadratic(PetscReal r)
{
  r = std::abs(r);
  if (r >= 1.5)
    return 0;
  return r < 0.5 ? 0.75 - r * r : (1.5 - r) * (1.5 - r) / 2;
}

// Exact integral in the track parameter, split at every knot of S1.
PetscReal average_linear(PetscReal node, PetscReal begin, PetscReal end)
{
  if (begin == end)
    return linear(node - begin);
  std::vector<PetscReal> cuts{0, 1};
  for (PetscReal knot : {node - 1, node, node + 1}) {
    const PetscReal t = (knot - begin) / (end - begin);
    if (0 < t && t < 1)
      cuts.push_back(t);
  }
  std::ranges::sort(cuts);
  PetscReal sum = 0;
  for (std::size_t i = 1; i < cuts.size(); i++) {
    const PetscReal lo = cuts[i - 1], hi = cuts[i];
    sum += (hi - lo) * (linear(node - begin - lo * (end - begin)) +
                        linear(node - begin - hi * (end - begin))) / 2;
  }
  return sum;
}

PetscReal transverse(PetscReal b, PetscReal c, const Vector3R& begin,
  const Vector3R& end, PetscInt axis_b, PetscInt axis_c)
{
  const PetscReal b0 = quadratic(b - begin[axis_b]);
  const PetscReal b1 = quadratic(b - end[axis_b]);
  const PetscReal c0 = quadratic(c - begin[axis_c]);
  const PetscReal c1 = quadratic(c - end[axis_c]);
  // S22 in (ef): endpoint pairs have weights 1/3, 1/6, 1/6, 1/3.
  return (b0 * c0 + b1 * c1) / 3 + (b0 * c1 + b1 * c0) / 6;
}

struct Track {
  const char* name;
  Vector3R begin, end;
};

struct Node {
  Vector3I g;
  Vector3R electric, magnetic0, magnetic1;
  PetscReal derivative[3][3]{};
};

std::vector<Node> reference(const World& world, const Track& track)
{
  std::vector<PetscReal> cuts{0, 1};
  for (Axis a : {X, Y, Z}) {
    const PetscReal lo = std::min(track.begin[a], track.end[a]);
    const PetscReal hi = std::max(track.begin[a], track.end[a]);
    for (PetscReal plane = std::floor(lo - 0.5) + 1.5;
         plane < hi; plane += 1) {
      if (plane > lo)
        cuts.push_back((plane - track.begin[a]) /
                       (track.end[a] - track.begin[a]));
    }
  }
  std::ranges::sort(cuts);
  cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
  std::vector<Node> nodes;
  for (PetscInt z = world.gstart[Z]; z < world.gend[Z]; z++)
    for (PetscInt y = world.gstart[Y]; y < world.gend[Y]; y++)
      for (PetscInt x = world.gstart[X]; x < world.gend[X]; x++) {
        Node n{};
        n.g = {x, y, z};
        bool outside = false;
        for (Axis a : {X, Y, Z})
          outside |= n.g[a] < std::min(track.begin[a], track.end[a]) - 2.5 ||
            n.g[a] > std::max(track.begin[a], track.end[a]) + 1.5;
        if (outside) {
          nodes.push_back(n);
          continue;
        }
        for (PetscInt component = 0; component < 3; component++) {
          n.magnetic0[component] = n.magnetic1[component] = 1;
          for (PetscInt a = 0; a < 3; a++) {
            const PetscReal pos = n.g[a] + (a == component ? 0 : 0.5);
            n.magnetic0[component] *= quadratic(pos - track.begin[a]);
            n.magnetic1[component] *= quadratic(pos - track.end[a]);
          }
        }
        for (std::size_t i = 1; i < cuts.size(); i++) {
          const PetscReal weight = cuts[i] - cuts[i - 1];
          const Vector3R begin = track.begin +
            (track.end - track.begin) * cuts[i - 1];
          const Vector3R end = track.begin +
            (track.end - track.begin) * cuts[i];
          for (PetscInt a = 0; a < 3; a++) {
            const PetscInt b = (a + 1) % 3, c = (a + 2) % 3;
            n.electric[a] += weight *
              average_linear(n.g[a] + 0.5, begin[a], end[a]) *
              transverse(n.g[b], n.g[c], begin, end, b, c);
            for (PetscInt component = 0; component < 3; component++) {
              n.derivative[a][component] += weight *
                average_linear(n.g[a] + (a == component ? 0.5 : 1),
                  begin[a], end[a]) *
                transverse(n.g[b] + (b == component ? 0 : 0.5),
                  n.g[c] + (c == component ? 0 : 0.5), begin, end, b, c);
            }
          }
        }
        nodes.push_back(n);
      }
  return nodes;
}

struct Checks {
  PetscReal max_error = 0;
  std::size_t cases = 0;

  PetscErrorCode compare(PetscReal actual, PetscReal expected,
    const std::string& name, const char* quantity, PetscInt component,
    const Vector3I& node = {})
  {
    PetscFunctionBeginUser;
    const PetscReal error = std::abs(actual - expected);
    max_error = std::max(max_error, error);

    PetscCheck(std::isfinite(actual) && std::isfinite(expected) &&
        error <= 1e-12 * (1 + std::abs(expected)),
      PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT,
      "%s: %s component %d node (%d,%d,%d): actual %.17g expected %.17g",
      name.c_str(), quantity, (int)component, (int)node[X], (int)node[Y],
      (int)node[Z], (double)actual, (double)expected);

    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

PetscErrorCode check_case(const World& world, const Track& track,
  const std::vector<Node>& nodes, const std::string& name,
  Vector3R*** e, Vector3R*** b, Vector3R*** j, Vector3R*** m,
  Checks& checks)
{
  PetscFunctionBeginUser;
  Vector3R expected_e, b0, b1;
  for (const auto& n : nodes) {
    const auto& g = n.g;
    j[g[Z]][g[Y]][g[X]] = {};
    m[g[Z]][g[Y]][g[X]] = {};
    for (PetscInt a = 0; a < 3; a++) {
      expected_e[a] += e[g[Z]][g[Y]][g[X]][a] * n.electric[a];
      b0[a] += b[g[Z]][g[Y]][g[X]][a] * n.magnetic0[a];
      b1[a] += b[g[Z]][g[Y]][g[X]][a] * n.magnetic1[a];
    }
  }
  const Vector3R mean_b = (b0.normalized() + b1.normalized()) / 2;
  const Vector3R hat_b = mean_b / mean_b.squared();
  Vector3R expected_grad, expected_rot;
  for (const auto& n : nodes) {
    const auto& g = n.g;
    PetscReal derivative[3][3]{};
    for (PetscInt a = 0; a < 3; a++) {
      Vector3I next = g;
      next[a]++;
      for (PetscInt component = 0; component < 3; component++) {
        if (n.derivative[a][component] == 0)
          continue;

        PetscCheck(next[a] < world.gend[a], PETSC_COMM_SELF,
          PETSC_ERR_ARG_OUTOFRANGE, "Reference stencil outside ghost grid");

        derivative[a][component] = n.derivative[a][component] *
          (b[next[Z]][next[Y]][next[X]][component] -
           b[g[Z]][g[Y]][g[X]][component]) / Dx[a];
        expected_grad[a] += hat_b[component] * derivative[a][component];
      }
    }
    expected_rot += Vector3R{
      derivative[Y][Z] - derivative[Z][Y],
      derivative[Z][X] - derivative[X][Z],
      derivative[X][Y] - derivative[Y][X]};
  }
  const auto physical = [](const Vector3R& r) {
    return Vector3R{r[X] * dx, r[Y] * dy, r[Z] * dz};
  };
  const Vector3R begin = physical(track.begin), end = physical(track.end);
  DriftKineticEsirkepov kernel(world, e, b, b, b, j, m);
  Vector3R ep, bp, grad, rot, point0, point1;
  PetscReal len;
  PetscCall(kernel.interpolate(ep, len, bp, grad, rot, end, begin));
  PetscCall(kernel.interpolate_B(point0, begin));
  PetscCall(kernel.interpolate_B(point1, end));
  // Nonzero components also expose all J weights on stationary axes.
  const Vector3R velocity{0.7, -1.1, 1.3};
  constexpr PetscReal charge = -0.8, mu = 0.6;
  PetscCall(kernel.decomposition(end, begin, velocity, charge, mu));
  PetscCall(checks.compare(len, (b0.length() + b1.length()) / 2,
    name, "|B|", 0));
  for (PetscInt a = 0; a < 3; a++) {
    PetscCall(checks.compare(ep[a], expected_e[a], name, "E", a));
    PetscCall(checks.compare(point0[a], b0[a], name, "B(begin)", a));
    PetscCall(checks.compare(point1[a], b1[a], name, "B(end)", a));
    PetscCall(checks.compare(bp[a], mean_b.normalized()[a], name, "b", a));
    PetscCall(checks.compare(grad[a], expected_grad[a], name, "gradB", a));
    PetscCall(checks.compare(rot[a], expected_rot[a], name, "rotB", a));
  }
  for (const auto& n : nodes) {
    const auto& g = n.g;
    for (PetscInt a = 0; a < 3; a++) {
      PetscCall(checks.compare(j[g[Z]][g[Y]][g[X]][a],
        charge * velocity[a] * n.electric[a], name, "J", a, g));
      PetscCall(checks.compare(m[g[Z]][g[Y]][g[X]][a],
        -mu * hat_b[a] * (n.magnetic0[a] + n.magnetic1[a]) / 2,
        name, "M", a, g));
    }
  }
  checks.cases++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode run()
{
  PetscFunctionBeginUser;
  World::set_geometry(8.0, 24.0, 32.0, 1.0, 0.5, 1.5, 2.0, 1.0, 1.0);
  World world;
  std::ranges::fill(world.bounds, DM_BOUNDARY_PERIODIC);
  PetscCall(world.initialize());
  Vec fields[4];
  for (auto& field : fields)
    PetscCall(DMCreateLocalVector(world.da, &field));
  Vector3R ***e, ***b, ***j, ***m;
  PetscCall(DMDAVecGetArray(world.da, fields[0], &e));
  PetscCall(DMDAVecGetArray(world.da, fields[1], &b));
  PetscCall(DMDAVecGetArray(world.da, fields[2], &j));
  PetscCall(DMDAVecGetArray(world.da, fields[3], &m));
  Vector3I center;
  for (Axis a : {X, Y, Z})
    center[a] = world.start[a] + world.size[a] / 2;
  const Vector3R origin{(PetscReal)center[X], (PetscReal)center[Y],
    (PetscReal)center[Z]};
  const std::vector<Track> tracks{
    {"stationary", {0.13, -0.27, 0.31}, {0.13, -0.27, 0.31}},
    {"stationary_node", {0, 0, 0}, {0, 0, 0}},
    {"stationary_face", {0.5, -0.5, 0.5}, {0.5, -0.5, 0.5}},
    {"axis", {-0.37, 0.11, 0.23}, {0.29, 0.11, 0.23}},
    {"plane", {-0.37, 0.28, 0.23}, {0.29, -0.41, 0.23}},
    {"cross_node", {-0.37, -0.19, -0.43}, {0.29, 0.41, 0.17}},
    {"same_side", {0.07, -0.42, 0.19}, {0.39, -0.13, 0.46}},
    {"start_node", {0, 0, 0}, {0.29, -0.41, 0.17}},
    {"faces", {-0.5, 0.5, -0.5}, {0.5, -0.5, 0.5}},
    {"tiny", {-3e-12, 0.21, -0.17}, {7e-12, 0.21, -0.17}},
    {"multiple_cells", {-1.1, 0.93, -0.71}, {1.17, -0.83, 1.09}},
  };
  Checks checks;
  for (const auto& source : tracks) {
    std::array<PetscInt, 3> permutation{0, 1, 2};
    do {
      for (PetscInt reverse = 0; reverse < 2; reverse++) {
        Track track{source.name, {}, {}};
        for (PetscInt a = 0; a < 3; a++) {
          track.begin[a] = origin[a] +
            (reverse ? source.end : source.begin)[permutation[a]];
          track.end[a] = origin[a] +
            (reverse ? source.begin : source.end)[permutation[a]];
        }
        const auto nodes = reference(world, track);
        const auto name = std::format("{}/{}{}{}/reverse{}", source.name,
          permutation[0], permutation[1], permutation[2], reverse);
        for (const auto& n : nodes) {
          const auto& g = n.g;
          const Vector3R r{(PetscReal)(g[X] - center[X]),
            (PetscReal)(g[Y] - center[Y]), (PetscReal)(g[Z] - center[Z])};
          for (PetscInt a = 0; a < 3; a++) {
            const PetscInt c = (a + 1) % 3;
            e[g[Z]][g[Y]][g[X]][a] =
              std::sin(0.7 * r[a] + 0.3 * r[c] + a) + 0.1 * r.squared();
            b[g[Z]][g[Y]][g[X]][a] = 2 + a +
              0.2 * std::cos(r[a] - 0.4 * r[c]) + 0.03 * r[a] * r[c];
          }
        }
        PetscCall(check_case(world, track, nodes, name, e, b, j, m, checks));
      }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  }
  // Basis fields isolate individual nodal weights and all derivative entries.
  for (PetscInt case_id : {5, 6}) {
    Track track = tracks[case_id];
    track.begin += origin;
    track.end += origin;
    const auto nodes = reference(world, track);
    for (PetscInt z = center[Z] - 2; z <= center[Z] + 2; z++)
      for (PetscInt y = center[Y] - 2; y <= center[Y] + 2; y++)
        for (PetscInt x = center[X] - 2; x <= center[X] + 2; x++)
          for (PetscInt a = 0; a < 3; a++) {
            for (const auto& n : nodes) {
              const auto& g = n.g;
              e[g[Z]][g[Y]][g[X]] = {};
              b[g[Z]][g[Y]][g[X]] = {2, 3, 4};
            }
            e[z][y][x][a] = 1;
            b[z][y][x][a] += 0.5;
            const auto name = std::format("{}/impulse/{},{},{}/{}",
              track.name, x, y, z, a);
            PetscCall(check_case(world, track, nodes, name, e, b, j, m, checks));
          }
  }
  PetscReal max_error;
  PetscCallMPI(MPI_Allreduce(&checks.max_error, &max_error, 1, MPIU_REAL,
    MPI_MAX, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "Checked %zu spline cases per rank; maximum absolute error %.3e.\n",
    checks.cases, (double)max_error));
  PetscCall(DMDAVecRestoreArray(world.da, fields[3], &m));
  PetscCall(DMDAVecRestoreArray(world.da, fields[2], &j));
  PetscCall(DMDAVecRestoreArray(world.da, fields[1], &b));
  PetscCall(DMDAVecRestoreArray(world.da, fields[0], &e));
  for (auto& field : fields)
    PetscCall(VecDestroy(&field));
  PetscCall(world.finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace

int main(int argc, char** argv)
{
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(run());
  PetscCall(PetscFinalize());
  return 0;
}
