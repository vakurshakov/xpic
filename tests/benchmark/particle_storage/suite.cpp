#include "suite.h"

#include <omp.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <list>
#include <random>

static constexpr char help[] =
  " Each step of every backend:                                  \n"
  " - snapshot (prev),                                           \n"
  " - push (Boris from prev),                                    \n"
  " - 2nd order deposit,                                         \n"
  " - update_cells (periodic wrap, migration, re-sort),          \n"
  " - optionally binary collisions (-collide k).                 \n"
  " Phases are timed apart. Options: -n -ppc -steps              \n"
  " -vth -omega -backends [lvabcde] -resort -collide.            \n"
  " Usage: [mpiexec -np N] ./particle_storage_suite.out [options]\n";

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  using namespace particle_storage;

  Config cfg;
  char backends[16] = "lvabcde";
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-xpic_tags", (PetscBool*)&xpic_tags, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-omega", &omega_dt, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vth", &vth, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-resort", &resort_period, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-buffer", &buffer_percent, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-stencil", &stencil_override, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-capfac", &capacity_factor_x10, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-collide", &collide_period, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-collide_index", (PetscBool*)&collide_index, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &cfg.n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ppc", &cfg.ppc, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-steps", &cfg.steps, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-backends", backends, sizeof(backends), NULL));

  int size;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  Local loc;
  PetscCall(create_local(cfg, 0, &loc));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "n=%d ppc=%d steps=%d ranks=%d threads=%d\n",
    cfg.n, cfg.ppc, cfg.steps, size, omp_get_max_threads()));

  const VecP init = make_particles(cfg, loc);
  for (const char* b = backends; *b; b++) {
    // a, b lose particles on several ranks, see README.md
    if (size > 1 && (*b == 'a' || *b == 'b'))
      continue;
    if (*b == 'l')
      PetscCall(run_list(cfg, loc, init));
    else if (*b == 'v')
      PetscCall(run_vectors(cfg, loc, init));
    else if (*b == 'a' || *b == 'b' || *b == 'c')
      PetscCall(run_swarm(cfg, init, *b));
    else if (*b == 'd')
      PetscCall(run_blocks(cfg, init));
    else if (*b == 'e')
      PetscCall(run_flat(cfg, init));
  }

  PetscCall(DMDestroy(&loc.da));
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

namespace particle_storage {

bool xpic_tags = false;
PetscReal omega_dt = 0.1;
PetscReal vth = 0.05;  // ~2% of the particles change cell per step
PetscInt resort_period = 0;
PetscInt buffer_percent = 5;
PetscInt stencil_override = -1;
PetscInt capacity_factor_x10 = 20;
PetscInt collide_period = 0;
bool collide_index = false;

PetscInt Local::cells() const
{
  return m[0] * m[1] * m[2];
}

PetscInt Local::cell(const PetscReal* r) const
{
  const PetscInt i = (PetscInt)std::floor(r[0]) - s[0];
  const PetscInt j = (PetscInt)std::floor(r[1]) - s[1];
  const PetscInt k = (PetscInt)std::floor(r[2]) - s[2];
  return (k * m[1] + j) * m[0] + i;
}


// Setup

// The cell DM of `DMSwarmSortGetAccess()` needs `sw >= 1`, which makes
// `DMLocatePoints()` accept the ghost cells and breaks the migration.
PetscErrorCode create_local(const Config& cfg, PetscInt sw, Local* loc)
{
  PetscFunctionBeginUser;
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
    DMDA_STENCIL_BOX, cfg.n, cfg.n, cfg.n, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, sw, NULL, NULL, NULL, &loc->da));
  PetscCall(DMSetFromOptions(loc->da));
  PetscCall(DMSetUp(loc->da));
  PetscCall(DMDASetUniformCoordinates(loc->da, 0, cfg.n, 0, cfg.n, 0, cfg.n));
  PetscCall(DMDAGetCorners(loc->da, &loc->s[0], &loc->s[1], &loc->s[2], &loc->m[0], &loc->m[1], &loc->m[2]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

VecP make_particles(const Config& cfg, const Local& loc)
{
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  std::mt19937 gen(12'345 + rank);
  std::uniform_real_distribution<PetscReal> uni(0, 1);
  std::normal_distribution<PetscReal> normal(0, vth);

  VecP init;
  init.reserve((size_t)loc.cells() * cfg.ppc);
  for (PetscInt k = 0; k < loc.m[2]; k++)
    for (PetscInt j = 0; j < loc.m[1]; j++)
      for (PetscInt i = 0; i < loc.m[0]; i++)
        for (PetscInt q = 0; q < cfg.ppc; q++)
          init.push_back(P{
            {
              loc.s[0] + i + uni(gen),
              loc.s[1] + j + uni(gen),
              loc.s[2] + k + uni(gen),
            },
            {
              normal(gen),
              normal(gen),
              normal(gen),
            },
          });
  return init;
}


// Kernels and checks shared by all backends

void wrap(PetscReal* r, PetscReal L)
{
  for (int a = 0; a < 3; a++) {
    if (r[a] < 0)
      r[a] += L;
    else if (r[a] >= L)
      r[a] -= L;
  }
}

void push(const PetscReal* rp, const PetscReal* vp,  //
  PetscReal* r, PetscReal* v, PetscReal L)
{
  static const PetscReal c = std::cos(omega_dt);
  static const PetscReal s = std::sin(omega_dt);

  const PetscReal Ex = E0 * std::sin(2.0 * M_PI * rp[0] / L);
  const PetscReal ux = vp[0] + 0.5 * Ex;
  v[0] = +c * ux + s * vp[1] + 0.5 * Ex;
  v[1] = -s * ux + c * vp[1];
  v[2] = vp[2];

  for (int a = 0; a < 3; a++)
    r[a] = rp[a] + v[a];
}

void deposit(PetscReal* rho, PetscInt n, const PetscReal* r)
{
  PetscInt i0[3];
  PetscReal w[3][3];
  for (int a = 0; a < 3; a++) {
    i0[a] = (PetscInt)std::round(r[a]);
    const PetscReal d = r[a] - i0[a];
    w[a][0] = 0.5 * (0.5 - d) * (0.5 - d);
    w[a][1] = 0.75 - d * d;
    w[a][2] = 0.5 * (0.5 + d) * (0.5 + d);
  }

  for (int k = 0; k < 3; k++) {
    const PetscInt kk = (((i0[2] + k - 1) % n) + n) % n;
    for (int j = 0; j < 3; j++) {
      const PetscInt jj = (((i0[1] + j - 1) % n) + n) % n;
      for (int i = 0; i < 3; i++) {
        const PetscInt ii = (((i0[0] + i - 1) % n) + n) % n;
#pragma omp atomic
        rho[(kk * n + jj) * n + ii] += w[0][i] * w[1][j] * w[2][k];
      }
    }
  }
}

// Deposited charge must equal the number of particles (weights sum to 1).
PetscErrorCode check_charge(const std::vector<PetscReal>& rho, PetscInt np)
{
  PetscFunctionBeginUser;
  PetscReal sum = 0;
  for (PetscReal x : rho)
    sum += x;

  PetscReal total_sum = 0;
  PetscInt total_n = 0;
  PetscCallMPI(MPI_Allreduce(&sum, &total_sum, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&np, &total_n, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

  PetscCheck(std::abs(total_sum - total_n) < 1e-9 * total_n, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Deposited charge %g differs from the particle number %d", (PetscReal)total_sum, total_n);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Collisions
//
// Coulomb-like scattering: the angle tan(theta / 2) = delta is normal with
// variance nu_dt = 0.05 (vth / |u|)^3, the relative velocity u is rotated by
// (theta, phi), Takizuka & Abe (1977). Cells are shuffled by an index array and
// paired in order; a cell is a counted range `[0, n)` addressed through
// `vel(i)`, so that all backends share the kernel.
static uint64_t splitmix64(uint64_t& state)
{
  uint64_t z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

static PetscReal uniform01(uint64_t& state)
{
  return (splitmix64(state) >> 11) * 0x1.0p-53;
}

// Returns |delta u|^2 of the pair
static PetscReal collide_pair(PetscReal* v1, PetscReal* v2, uint64_t& rng)
{
  const PetscReal u[3] = {v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]};
  const PetscReal u_perp2 = u[0] * u[0] + u[1] * u[1];
  const PetscReal u2 = u_perp2 + u[2] * u[2];
  if (u2 == 0)
    return 0;

  const PetscReal u_abs = std::sqrt(u2);
  const PetscReal u_perp = std::sqrt(u_perp2);
  const PetscReal sigma = std::sqrt(0.05 * vth * vth * vth / (u2 * u_abs));

  const PetscReal delta = sigma *
    std::sqrt(-2.0 * std::log(1.0 - uniform01(rng))) *
    std::cos(2.0 * M_PI * uniform01(rng));
  const PetscReal sin_theta = 2.0 * delta / (1.0 + delta * delta);
  const PetscReal one_minus_cos = 2.0 * delta * delta / (1.0 + delta * delta);
  const PetscReal phi = 2.0 * M_PI * uniform01(rng);
  const PetscReal cos_phi = std::cos(phi);
  const PetscReal sin_phi = std::sin(phi);

  PetscReal du[3];
  if (u_perp > 1e-12 * u_abs) {
    du[0] =
      (u[0] * u[2] * sin_theta * cos_phi - u[1] * u_abs * sin_theta * sin_phi) /
        u_perp -
      u[0] * one_minus_cos;
    du[1] =
      (u[1] * u[2] * sin_theta * cos_phi + u[0] * u_abs * sin_theta * sin_phi) /
        u_perp -
      u[1] * one_minus_cos;
    du[2] = -u_perp * sin_theta * cos_phi - u[2] * one_minus_cos;
  }
  else {
    du[0] = u_abs * sin_theta * cos_phi;
    du[1] = u_abs * sin_theta * sin_phi;
    du[2] = -u[2] * one_minus_cos;
  }

  for (int a = 0; a < 3; a++) {
    v1[a] += 0.5 * du[a];
    v2[a] -= 0.5 * du[a];
  }
  return du[0] * du[0] + du[1] * du[1] + du[2] * du[2];
}

template<class Velocity>
static PetscReal collide_cell(
  PetscInt n, Velocity&& vel, uint64_t seed, std::vector<PetscInt>& idx)
{
  idx.resize(n);
  for (PetscInt i = 0; i < n; i++)
    idx[i] = i;
  for (PetscInt i = n - 1; i > 0; i--)
    std::swap(idx[i], idx[splitmix64(seed) % (i + 1)]);

  PetscReal changed = 0;
  for (PetscInt k = 0; k + 1 < n; k += 2)
    changed += collide_pair(vel(idx[k]), vel(idx[k + 1]), seed);
  return changed;
}

static uint64_t cell_seed(PetscInt step, PetscInt cell)
{
  return ((uint64_t)step << 32) ^ (uint64_t)cell;
}

void Moments::add(const PetscReal* v)
{
  for (int a = 0; a < 3; a++) {
    p[a] += v[a];
    e += v[a] * v[a];
  }
}

PetscErrorCode check_collisions(
  const Moments& before, const Moments& after, PetscReal changed)
{
  PetscFunctionBeginUser;
  for (int a = 0; a < 3; a++) {
    PetscCheck(std::abs(after.p[a] - before.p[a]) < 1e-8, PETSC_COMM_SELF, PETSC_ERR_USER,
      "Collisions changed momentum[%d]: %g -> %g", a, (double)before.p[a], (double)after.p[a]);
  }
  PetscCheck(std::abs(after.e - before.e) < 1e-10 * before.e, PETSC_COMM_SELF, PETSC_ERR_USER,
    "Collisions changed energy: %.15g -> %.15g", (double)before.e, (double)after.e);
  PetscCheck(changed > 0, PETSC_COMM_SELF, PETSC_ERR_USER, "Collisions did nothing");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode report(const char* name, const Phases& t, const Config& cfg)
{
  PetscFunctionBeginUser;
  const PetscReal arr[6]{
    t.prepare,
    t.push,
    t.deposit,
    t.migrate,
    t.sort,
    t.collide,
  };
  PetscReal max[6];
  PetscCallMPI(MPI_Allreduce(arr, max, 6, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));

  const PetscReal s = 1e3 / cfg.steps;  // ms per step
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-8s prepare %8.3f  push %8.3f  deposit %8.3f  migrate %8.3f  sort %8.3f  collide %8.3f  total %8.3f  [ms/step]\n",
    name, max[0] * s, max[1] * s, max[2] * s, max[3] * s, max[4] * s, max[5] * s, (max[0] + max[1] + max[2] + max[3] + max[4] + max[5]) * s));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// Neighbour slot of a (wrapped) position, same layout as `World::neighbors`.
int slot_of(const Local& loc, const PetscReal* r)
{
  int slot = 0;
  for (int a = 2; a >= 0; a--) {
    const PetscInt c = (PetscInt)std::floor(r[a]);
    slot = slot * 3 + (c < loc.s[a] ? 0 : (c < loc.s[a] + loc.m[a] ? 1 : 2));
  }
  return slot;
}

// The sender's slot s is the receiver's slot 26 - s. With 2 ranks several
// slots share the same neighbour, so the pairing is explicit.
PetscErrorCode exchange(const PetscMPIInt* neighbors,
  const VecP (&outgoing)[slots], VecP (&incoming)[slots])
{
  PetscFunctionBeginUser;
  auto send_tag = [&](int s, int base)
  {
    return xpic_tags ? base : base + s;
  };

  auto recv_tag = [&](int s, int base)
  {
    return xpic_tags ? base : base + slots - 1 - s;
  };

  size_t o_num[slots];
  size_t i_num[slots];
  MPI_Request reqs[2 * slots];
  int req = 0;
  for (int s = 0; s < slots; s++) {
    o_num[s] = outgoing[s].size();
    if (s == center)
      continue;
    PetscCallMPI(MPI_Isend(&o_num[s], 1, MPIU_SIZE_T, neighbors[s], send_tag(s, 100), PETSC_COMM_WORLD, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(&i_num[s], 1, MPIU_SIZE_T, neighbors[s], recv_tag(s, 100), PETSC_COMM_WORLD, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));

  req = 0;
  for (int s = 0; s < slots; s++) {
    if (s == center)
      continue;
    incoming[s].resize(i_num[s]);
    PetscCallMPI(MPI_Isend(outgoing[s].data(), o_num[s] * sizeof(P), MPI_BYTE, neighbors[s], send_tag(s, 200), PETSC_COMM_WORLD, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(incoming[s].data(), i_num[s] * sizeof(P), MPI_BYTE, neighbors[s], recv_tag(s, 200), PETSC_COMM_WORLD, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// Backend l
PetscErrorCode run_list(const Config& cfg, const Local& loc, const VecP& init)
{
  PetscFunctionBeginUser;
  const PetscInt n = cfg.n;
  const PetscReal L = n;
  const PetscInt Ncells = loc.cells();
  const PetscMPIInt* neighbors;
  PetscCall(DMDAGetNeighbors(loc.da, &neighbors));

  std::vector<std::list<P>> storage(Ncells);
  std::vector<std::vector<P>> prev(Ncells);
  std::vector<PetscReal> rho((size_t)n * n * n);

  for (const P& p : init)
    storage[loc.cell(p.r)].push_back(p);

  Phases t;

  for (PetscInt step = 0; step < cfg.steps; step++) {
    PetscReal t0 = omp_get_wtime();

#pragma omp parallel for
    for (PetscInt g = 0; g < Ncells; g++) {
      prev[g].assign(storage[g].begin(), storage[g].end());
    }
    PetscReal t1 = omp_get_wtime();
    t.prepare += t1 - t0;

#pragma omp parallel for
    for (PetscInt g = 0; g < Ncells; g++) {
      for (size_t i = 0; P& p : storage[g]) {
        push(prev[g][i].r, prev[g][i].v, p.r, p.v, L);
        i++;
      }
    }
    PetscReal t2 = omp_get_wtime();
    t.push += t2 - t1;

    std::fill(rho.begin(), rho.end(), 0);

#pragma omp parallel for
    for (PetscInt g = 0; g < Ncells; g++) {
      for (const P& p : storage[g])
        deposit(rho.data(), n, p.r);
    }
    PetscReal t3 = omp_get_wtime();
    t.deposit += t3 - t2;

    PetscInt np = 0;
    for (const auto& cell : storage)
      np += (PetscInt)cell.size();
    PetscCall(check_charge(rho, np));

    // update_cells: local moves, then the neighbour exchange as in
    // `interfaces::Particles::update_cells_mpi()`
    t3 = omp_get_wtime();
    VecP outgoing[slots];
    VecP incoming[slots];
    for (PetscInt g = 0; g < Ncells; g++) {
      for (auto it = storage[g].begin(); it != storage[g].end();) {
        wrap(it->r, L);
        const int slot = slot_of(loc, it->r);

        if (slot == center) {
          const PetscInt ng = loc.cell(it->r);
          if (ng == g) {
            it++;
            continue;
          }
          storage[ng].emplace_back(std::move(*it));
        }
        else {
          outgoing[slot].push_back(*it);
        }
        it = storage[g].erase(it);
      }
    }

    PetscCall(exchange(neighbors, outgoing, incoming));

    for (int s = 0; s < slots; s++) {
      for (const P& p : incoming[s])
        storage[loc.cell(p.r)].push_back(p);
    }
    PetscReal t4 = omp_get_wtime();
    t.sort += t4 - t3;
  }

  PetscCall(report("list", t, cfg));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// Backend v: as the list, but `vector<vector<P>>` with swap-with-last removal
PetscErrorCode run_vectors(const Config& cfg, const Local& loc, const VecP& init)
{
  PetscFunctionBeginUser;
  const PetscInt n = cfg.n;
  const PetscReal L = n;
  const PetscInt Ncells = loc.cells();
  const PetscMPIInt* neighbors;
  PetscCall(DMDAGetNeighbors(loc.da, &neighbors));

  std::vector<VecP> storage(Ncells);
  std::vector<VecP> prev(Ncells);
  std::vector<PetscReal> rho((size_t)n * n * n);

  for (const P& p : init)
    storage[loc.cell(p.r)].push_back(p);

  Phases t;
  for (PetscInt step = 0; step < cfg.steps; step++) {
    PetscReal t0 = omp_get_wtime();

#pragma omp parallel for
    for (PetscInt g = 0; g < Ncells; g++) {
      prev[g].assign(storage[g].begin(), storage[g].end());
    }
    PetscReal t1 = omp_get_wtime();
    t.prepare += t1 - t0;

#pragma omp parallel for
    for (PetscInt g = 0; g < Ncells; g++) {
      for (size_t i = 0; P& p : storage[g]) {
        push(prev[g][i].r, prev[g][i].v, p.r, p.v, L);
        i++;
      }
    }
    PetscReal t2 = omp_get_wtime();
    t.push += t2 - t1;

    std::fill(rho.begin(), rho.end(), 0);

#pragma omp parallel for
    for (PetscInt g = 0; g < Ncells; g++) {
      for (const P& p : storage[g])
        deposit(rho.data(), n, p.r);
    }
    PetscReal t3 = omp_get_wtime();
    t.deposit += t3 - t2;

    PetscInt np = 0;
    for (const auto& cell : storage)
      np += (PetscInt)cell.size();
    PetscCall(check_charge(rho, np));

    // update_cells: local moves, then the neighbour exchange as in
    // `interfaces::Particles::update_cells_mpi()`
    t3 = omp_get_wtime();
    VecP outgoing[slots];
    VecP incoming[slots];
    for (PetscInt g = 0; g < Ncells; g++) {
      auto& cell = storage[g];
      for (size_t i = 0; i < cell.size();) {
        wrap(cell[i].r, L);
        const int slot = slot_of(loc, cell[i].r);
        if (slot == center) {
          const PetscInt ng = loc.cell(cell[i].r);
          if (ng == g) {
            i++;
            continue;
          }
          storage[ng].push_back(cell[i]);
        }
        else {
          outgoing[slot].push_back(cell[i]);
        }
        cell[i] = cell.back();
        cell.pop_back();
      }
    }

    PetscCall(exchange(neighbors, outgoing, incoming));

    for (int s = 0; s < slots; s++) {
      for (const P& p : incoming[s])
        storage[loc.cell(p.r)].push_back(p);
    }
    PetscReal t4 = omp_get_wtime();
    t.sort += t4 - t3;

    if (collide_period > 0 && (step + 1) % collide_period == 0) {
      Moments before, after;
      for (const VecP& cell : storage)
        for (const P& p : cell)
          before.add(p.v);

      PetscReal changed = 0;
#pragma omp parallel reduction(+ : changed)
      {
        std::vector<PetscInt> idx;
#pragma omp for
        for (PetscInt g = 0; g < Ncells; g++) {
          changed += collide_cell((PetscInt)storage[g].size(),
            [&](PetscInt i)
            {
              return storage[g][i].v;
            },
            cell_seed(step, g), idx);
        }
      }
      t.collide += omp_get_wtime() - t4;

      for (const VecP& cell : storage)
        for (const P& p : cell)
          after.add(p.v);
      PetscCall(check_collisions(before, after, changed));
    }
  }

  PetscCall(report("vector", t, cfg));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// Backends a, b, c
//
// a: cells via `DMSwarmSortGetAccess()` + `DMSwarmSortGetPointsPerCell()`;
// b: swarm as a field registry + `DMSwarmMigrate()`, own counting sort by cell,
//    raw `DMSwarmGetField()` arrays inside the OpenMP region;
// c: as b, but our own neighbour exchange instead of `DMSwarmMigrate()`.
PetscErrorCode run_swarm(const Config& cfg, const VecP& init, char mode)
{
  PetscFunctionBeginUser;
  const bool own_sort = mode != 'a';
  const bool own_migrate = mode == 'c';
  Local loc;
  PetscCall(create_local(cfg, stencil_override >= 0 ? stencil_override : (own_sort ? 0 : 1), &loc));

  const PetscReal L = cfg.n;
  const PetscInt n = cfg.n;
  const PetscInt total = (PetscInt)init.size();
  PetscInt expected;
  PetscCallMPI(MPI_Allreduce(&total, &expected, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

  DM sw;
  PetscCall(create_swarm(init, loc, &sw));
  // Initial placement is already local; DMSwarmMigrate() is the backend a/b path
  if (!own_migrate)
    PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
  PetscCall(diagnose_cells(sw, cfg));

  PetscInt ncells_sw = 0;
  std::vector<PetscInt> offsets;

  if (own_sort)
    PetscCall(counting_sort(sw, loc, offsets));
  else {
    PetscCall(DMSwarmSortGetAccess(sw));
    const PetscInt* element;
    PetscInt nel, npe;
    PetscCall(DMDAGetElements(loc.da, &nel, &npe, &element));
    ncells_sw = nel;
    PetscCall(DMDARestoreElements(loc.da, &nel, &npe, &element));
  }

  std::vector<PetscReal> rho((size_t)n * n * n);
  Phases t;
  for (PetscInt step = 0; step < cfg.steps; step++) {
    PetscReal t0 = omp_get_wtime();
    PetscCall(snapshot(sw));

    PetscReal t1 = omp_get_wtime();
    t.prepare += t1 - t0;

    Attributes f;
    PetscCall(get_attributes(sw, f));

    if (own_sort) {
#pragma omp parallel for
      for (PetscInt c = 0; c < loc.cells(); c++) {
        for (PetscInt i = offsets[c]; i < offsets[c + 1]; i++)
          push(&f.rp[3 * i], &f.vp[3 * i], &f.r[3 * i], &f.v[3 * i], L);
      }
    }
    else {
      for (PetscInt c = 0; c < ncells_sw; c++) {
        PetscInt np, *ids;
        PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &np, &ids));
#pragma omp parallel for
        for (PetscInt q = 0; q < np; q++) {
          const PetscInt i = ids[q];
          push(&f.rp[3 * i], &f.vp[3 * i], &f.r[3 * i], &f.v[3 * i], L);
        }
        PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &np, &ids));
      }
    }
    PetscReal t2 = omp_get_wtime();
    t.push += t2 - t1;

    std::fill(rho.begin(), rho.end(), 0);

    if (own_sort) {
#pragma omp parallel for
      for (PetscInt c = 0; c < loc.cells(); c++) {
        for (PetscInt i = offsets[c]; i < offsets[c + 1]; i++)
          deposit(rho.data(), n, &f.r[3 * i]);
      }
    }
    else {
      for (PetscInt c = 0; c < ncells_sw; c++) {
        PetscInt np, *ids;
        PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &np, &ids));
#pragma omp parallel for
        for (PetscInt q = 0; q < np; q++)
          deposit(rho.data(), n, &f.r[3 * ids[q]]);
        PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &np, &ids));
      }
    }
    PetscReal t3 = omp_get_wtime();
    t.deposit += t3 - t2;

    PetscInt np = f.np;
    PetscCall(restore_attributes(sw, f));
    PetscCall(check_charge(rho, np));

    if (!own_sort)
      PetscCall(DMSwarmSortRestoreAccess(sw));

    if (own_migrate)
      PetscCall(migrate_own(sw, loc, L, t));
    else
      PetscCall(wrap_and_migrate(sw, L, t));

    PetscCall(check_count(sw, expected));

    t3 = omp_get_wtime();
    if (own_sort)
      PetscCall(counting_sort(sw, loc, offsets));
    else
      PetscCall(DMSwarmSortGetAccess(sw));
    PetscReal t4 = omp_get_wtime();
    t.sort += t4 - t3;

    // `c` is sorted by cell after every step, collisions only need the offsets
    if (mode == 'c' && collide_period > 0 && (step + 1) % collide_period == 0) {
      PetscCall(collide_swarm(sw, offsets, NULL, step));
      t.collide += omp_get_wtime() - t4;
    }
  }

  if (!own_sort)
    PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&loc.da));
  PetscCall(report(mode == 'a' ? "swarm_a" : (mode == 'b' ? "swarm_b" : "swarm_c"), t, cfg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode create_swarm(const VecP& init, const Local& loc, DM* sw)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(PETSC_COMM_WORLD, sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, 3));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, loc.da));
  PetscCall(DMSwarmSetMigrateType(*sw, DMSWARM_MIGRATE_DMCELLNSCATTER));
  PetscCall(DMSwarmInitializeFieldRegister(*sw));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "v", 3, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "r_prev", 3, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "v_prev", 3, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMSwarmSetLocalSizes(*sw, (PetscInt)init.size(), (PetscInt)init.size() * buffer_percent / 100));

  Attributes f;
  PetscCall(get_attributes(*sw, f));
  for (PetscInt i = 0; i < f.np; i++) {
    std::memcpy(&f.r[3 * i], init[i].r, 3 * sizeof(PetscReal));
    std::memcpy(&f.v[3 * i], init[i].v, 3 * sizeof(PetscReal));
  }
  PetscCall(restore_attributes(*sw, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode diagnose_cells(DM sw, const Config& cfg)
{
  PetscFunctionBeginUser;
  const PetscInt n = cfg.n;

  Attributes f;
  PetscCall(get_attributes(sw, f));

  std::vector<PetscInt> count((size_t)n * n * n), total((size_t)n * n * n);
  for (PetscInt i = 0; i < f.np; i++) {
    const PetscReal* r = &f.r[3 * i];
    count[((PetscInt)r[2] * n + (PetscInt)r[1]) * n + (PetscInt)r[0]]++;
  }

  PetscCall(restore_attributes(sw, f));
  PetscCallMPI(MPI_Allreduce(count.data(), total.data(), n * n * n, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

  PetscInt bad = 0;
  PetscInt lo[3] = {n, n, n};
  PetscInt hi[3] = {-1, -1, -1};
  for (PetscInt c = 0; c < n * n * n; c++) {
    if (total[c] == cfg.ppc)
      continue;
    const PetscInt idx[3] = {c % n, (c / n) % n, c / (n * n)};
    bad++;
    for (int a = 0; a < 3; a++) {
      lo[a] = std::min(lo[a], idx[a]);
      hi[a] = std::max(hi[a], idx[a]);
    }
  }
  if (bad)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "diag: %d cells with count != ppc, x[%d,%d] y[%d,%d] z[%d,%d]\n",
      bad, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode wrap_and_migrate(DM sw, PetscReal L, Phases& t)
{
  PetscFunctionBeginUser;
  PetscReal t0 = omp_get_wtime();

  Attributes f;
  PetscCall(get_attributes(sw, f));

#pragma omp parallel for
  for (PetscInt i = 0; i < f.np; i++)
    wrap(&f.r[3 * i], L);

  PetscCall(restore_attributes(sw, f));
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));

  t.migrate += omp_get_wtime() - t0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Local particles are compacted in place, the leaving ones are packed, the
// swarm is resized once and the incoming ones are appended.
PetscErrorCode migrate_own(DM sw, const Local& loc, PetscReal L, Phases& t)
{
  PetscFunctionBeginUser;
  PetscReal t0 = omp_get_wtime();

  const PetscMPIInt* neighbors;
  PetscCall(DMDAGetNeighbors(loc.da, &neighbors));

  Attributes f;
  PetscCall(get_attributes(sw, f));

  VecP outgoing[slots];
  VecP incoming[slots];
  PetscInt kept = 0;
  for (PetscInt i = 0; i < f.np; i++) {
    wrap(&f.r[3 * i], L);
    const int slot = slot_of(loc, &f.r[3 * i]);
    if (slot == center) {
      if (kept != i) {
        std::memcpy(&f.r[3 * kept], &f.r[3 * i], 3 * sizeof(PetscReal));
        std::memcpy(&f.v[3 * kept], &f.v[3 * i], 3 * sizeof(PetscReal));
      }
      kept++;
    }
    else {
      P p;
      std::memcpy(p.r, &f.r[3 * i], 3 * sizeof(PetscReal));
      std::memcpy(p.v, &f.v[3 * i], 3 * sizeof(PetscReal));
      outgoing[slot].push_back(p);
    }
  }
  PetscCall(restore_attributes(sw, f));

  PetscCall(exchange(neighbors, outgoing, incoming));

  PetscInt received = 0;
  for (const auto& in : incoming)
    received += (PetscInt)in.size();
  PetscCall(DMSwarmSetLocalSizes(sw, kept + received, -1));

  PetscCall(get_attributes(sw, f));
  PetscInt w = kept;
  for (const auto& in : incoming) {
    for (const P& p : in) {
      std::memcpy(&f.r[3 * w], p.r, 3 * sizeof(PetscReal));
      std::memcpy(&f.v[3 * w], p.v, 3 * sizeof(PetscReal));
      w++;
    }
  }
  PetscCall(restore_attributes(sw, f));

  t.migrate += omp_get_wtime() - t0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode snapshot(DM sw)
{
  PetscFunctionBeginUser;
  Attributes f;
  PetscCall(get_attributes(sw, f));
#pragma omp parallel for
  for (PetscInt i = 0; i < f.np; i++) {
    std::memcpy(&f.rp[3 * i], &f.r[3 * i], 3 * sizeof(PetscReal));
    std::memcpy(&f.vp[3 * i], &f.v[3 * i], 3 * sizeof(PetscReal));
  }
  PetscCall(restore_attributes(sw, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_count(DM sw, PetscInt expected)
{
  PetscFunctionBeginUser;
  PetscInt np, total;
  PetscCall(DMSwarmGetLocalSize(sw, &np));
  PetscCallMPI(MPI_Allreduce(&np, &total, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCheck(total == expected, PETSC_COMM_WORLD, PETSC_ERR_USER, "Particles lost: %d of %d left", total, expected);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Counting sort by local cell: physically permutes all the fields.
PetscErrorCode counting_sort(
  DM sw, const Local& loc, std::vector<PetscInt>& offsets)
{
  PetscFunctionBeginUser;
  Attributes f;
  PetscCall(get_attributes(sw, f));

  const PetscInt nc = loc.cells();
  offsets.assign(nc + 1, 0);
  std::vector<PetscInt> key(f.np), dst(f.np);
  for (PetscInt i = 0; i < f.np; i++) {
    key[i] = loc.cell(&f.r[3 * i]);
    PetscCheck(key[i] >= 0 && key[i] < nc, PETSC_COMM_SELF, PETSC_ERR_USER, "Particle %d is outside of the local box after migration", i);
    offsets[key[i] + 1]++;
  }
  for (PetscInt c = 0; c < nc; c++)
    offsets[c + 1] += offsets[c];

  std::vector<PetscInt> fill(offsets.begin(), offsets.end() - 1);
  for (PetscInt i = 0; i < f.np; i++)
    dst[i] = fill[key[i]]++;

  // `r_prev` and `v_prev` are rewritten by the next snapshot: not permuted
  std::vector<PetscReal> scratch(3 * (size_t)f.np);
  for (PetscReal* field : {f.r, f.v}) {
#pragma omp parallel for
    for (PetscInt i = 0; i < f.np; i++)
      std::memcpy(&scratch[3 * dst[i]], &field[3 * i], 3 * sizeof(PetscReal));
    std::memcpy(field, scratch.data(), 3 * (size_t)f.np * sizeof(PetscReal));
  }
  PetscCall(restore_attributes(sw, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// As `counting_sort()`, but only the indices `perm[offsets[c]..offsets[c + 1])`
// of the particles of cell c are produced, the fields are not permuted.
PetscErrorCode build_index(DM sw, const Local& loc,
  std::vector<PetscInt>& offsets, std::vector<PetscInt>& perm)
{
  PetscFunctionBeginUser;
  Attributes f;
  PetscCall(get_attributes(sw, f));

  const PetscInt nc = loc.cells();
  offsets.assign(nc + 1, 0);
  std::vector<PetscInt> key(f.np);
  perm.resize(f.np);
  for (PetscInt i = 0; i < f.np; i++) {
    key[i] = loc.cell(&f.r[3 * i]);
    PetscCheck(key[i] >= 0 && key[i] < nc, PETSC_COMM_SELF, PETSC_ERR_USER, "Particle %d is outside of the local box after migration", i);
    offsets[key[i] + 1]++;
  }
  for (PetscInt c = 0; c < nc; c++)
    offsets[c + 1] += offsets[c];

  std::vector<PetscInt> fill(offsets.begin(), offsets.end() - 1);
  for (PetscInt i = 0; i < f.np; i++)
    perm[fill[key[i]]++] = i;

  PetscCall(restore_attributes(sw, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Cell c is `[offsets[c], offsets[c + 1])` of the fields, or of `perm` if given
PetscErrorCode collide_swarm(DM sw, const std::vector<PetscInt>& offsets,
  const PetscInt* perm, PetscInt step)
{
  PetscFunctionBeginUser;
  Attributes f;
  PetscCall(get_attributes(sw, f));

  Moments before, after;
  for (PetscInt i = 0; i < f.np; i++)
    before.add(&f.v[3 * i]);

  const PetscInt nc = (PetscInt)offsets.size() - 1;
  PetscReal changed = 0;
#pragma omp parallel reduction(+ : changed)
  {
    std::vector<PetscInt> idx;
#pragma omp for
    for (PetscInt c = 0; c < nc; c++) {
      const PetscInt first = offsets[c];
      changed += collide_cell(
        offsets[c + 1] - first,
        [&](PetscInt i)
        {
          return &f.v[3 * (perm ? perm[first + i] : first + i)];
        },
        cell_seed(step, c), idx);
    }
  }

  for (PetscInt i = 0; i < f.np; i++)
    after.add(&f.v[3 * i]);
  PetscCall(restore_attributes(sw, f));
  PetscCall(check_collisions(before, after, changed));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode get_attributes(DM sw, Attributes& f)
{
  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetLocalSize(sw, &f.np));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void**)&f.r));
  PetscCall(DMSwarmGetField(sw, "v", NULL, NULL, (void**)&f.v));
  PetscCall(DMSwarmGetField(sw, "r_prev", NULL, NULL, (void**)&f.rp));
  PetscCall(DMSwarmGetField(sw, "v_prev", NULL, NULL, (void**)&f.vp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode restore_attributes(DM sw, Attributes& f)
{
  PetscFunctionBeginUser;
  PetscCall(DMSwarmRestoreField(sw, "v_prev", NULL, NULL, (void**)&f.vp));
  PetscCall(DMSwarmRestoreField(sw, "r_prev", NULL, NULL, (void**)&f.rp));
  PetscCall(DMSwarmRestoreField(sw, "v", NULL, NULL, (void**)&f.v));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void**)&f.r));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// Backend d: the swarm fields hold fixed-capacity blocks, one per cell, with
// a per-cell count (the array analogue of the list's `vector<list<P>>`).
// Only the particles that change cell are moved: a leaving particle is
// replaced by the last one of its block, an arriving one is appended to the
// destination block. A full block triggers a rebuild with a larger capacity.
PetscErrorCode run_blocks(const Config& cfg, const VecP& init)
{
  PetscFunctionBeginUser;
  Local loc;
  PetscCall(create_local(cfg, 0, &loc));

  const PetscReal L = cfg.n;
  const PetscInt n = cfg.n;
  const PetscInt total = (PetscInt)init.size();
  PetscInt expected;
  PetscCallMPI(MPI_Allreduce(&total, &expected, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

  DM sw;
  Blocks b;
  PetscCall(create_swarm(init, loc, &sw));
  PetscCall(fill_blocks(sw, loc, b, init, cfg.ppc * capacity_factor_x10 / 10));

  std::vector<PetscReal> rho((size_t)n * n * n);
  Phases t;
  for (PetscInt step = 0; step < cfg.steps; step++) {
    PetscReal t0 = omp_get_wtime();

    Attributes f;
    PetscCall(get_attributes(sw, f));

#pragma omp parallel for
    for (PetscInt c = 0; c < loc.cells(); c++) {
      const PetscInt i = c * b.cap;
      std::memcpy(&f.rp[3 * i], &f.r[3 * i], 3 * b.cnt[c] * sizeof(PetscReal));
      std::memcpy(&f.vp[3 * i], &f.v[3 * i], 3 * b.cnt[c] * sizeof(PetscReal));
    }
    PetscReal t1 = omp_get_wtime();
    t.prepare += t1 - t0;

#pragma omp parallel for
    for (PetscInt c = 0; c < loc.cells(); c++) {
      for (PetscInt i = c * b.cap; i < c * b.cap + b.cnt[c]; i++)
        push(&f.rp[3 * i], &f.vp[3 * i], &f.r[3 * i], &f.v[3 * i], L);
    }
    PetscReal t2 = omp_get_wtime();
    t.push += t2 - t1;

    std::fill(rho.begin(), rho.end(), 0);
#pragma omp parallel for
    for (PetscInt c = 0; c < loc.cells(); c++) {
      for (PetscInt i = c * b.cap; i < c * b.cap + b.cnt[c]; i++)
        deposit(rho.data(), n, &f.r[3 * i]);
    }
    t.deposit += omp_get_wtime() - t2;

    PetscCall(restore_attributes(sw, f));

    PetscInt np = 0;
    for (PetscInt c : b.cnt)
      np += c;
    PetscCall(check_charge(rho, np));

    PetscCall(update_blocks(sw, loc, b, L, t));
    np = 0;
    for (PetscInt c : b.cnt)
      np += c;
    PetscInt sum;
    PetscCallMPI(MPI_Allreduce(&np, &sum, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscCheck(sum == expected, PETSC_COMM_WORLD, PETSC_ERR_USER, "Particles lost: %d of %d left", sum, expected);
  }

  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&loc.da));
  PetscCall(report("swarm_d", t, cfg));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "         block capacity %d, rebuilds (rank 0) %d\n", b.cap, b.rebuilds));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode update_blocks(
  DM sw, const Local& loc, Blocks& b, PetscReal L, Phases& t)
{
  PetscFunctionBeginUser;
  PetscReal t0 = omp_get_wtime();

  const PetscMPIInt* neighbors;
  PetscCall(DMDAGetNeighbors(loc.da, &neighbors));

  VecP outgoing[slots], incoming[slots];
  VecP arrivals;  // Particles that stay on this rank, with new cells

  Attributes f;
  PetscCall(get_attributes(sw, f));

  const PetscInt nc = loc.cells();
#pragma omp parallel
  {
    VecP out_l[slots], arrivals_l;
#pragma omp for nowait
    for (PetscInt c = 0; c < nc; c++) {
      for (PetscInt q = 0; q < b.cnt[c];) {
        const PetscInt i = c * b.cap + q;
        wrap(&f.r[3 * i], L);
        const int slot = slot_of(loc, &f.r[3 * i]);
        if (slot == center && loc.cell(&f.r[3 * i]) == c) {
          q++;
          continue;
        }

        P p;
        std::memcpy(p.r, &f.r[3 * i], 3 * sizeof(PetscReal));
        std::memcpy(p.v, &f.v[3 * i], 3 * sizeof(PetscReal));
        (slot == center ? arrivals_l : out_l[slot]).push_back(p);

        const PetscInt last = c * b.cap + --b.cnt[c];
        if (i != last) {
          std::memcpy(&f.r[3 * i], &f.r[3 * last], 3 * sizeof(PetscReal));
          std::memcpy(&f.v[3 * i], &f.v[3 * last], 3 * sizeof(PetscReal));
        }
      }
    }
#pragma omp critical
    {
      for (int s = 0; s < slots; s++)
        outgoing[s].insert(outgoing[s].end(), out_l[s].begin(), out_l[s].end());
      arrivals.insert(arrivals.end(), arrivals_l.begin(), arrivals_l.end());
    }
  }

  PetscCall(restore_attributes(sw, f));
  PetscCall(exchange(neighbors, outgoing, incoming));

  PetscReal t1 = omp_get_wtime();
  t.migrate += t1 - t0;

  for (const auto& in : incoming)
    arrivals.insert(arrivals.end(), in.begin(), in.end());

  std::vector<PetscInt> added(nc, 0);
  bool overflow = false;
  for (const P& p : arrivals) {
    const PetscInt c = loc.cell(p.r);
    overflow |= b.cnt[c] + ++added[c] > b.cap;
  }

  if (overflow) {
    PetscCall(gather_blocks(sw, b, arrivals));  // Blocks and arrivals together
    PetscCall(fill_blocks(sw, loc, b, arrivals, 2 * b.cap));
    b.rebuilds++;
  }
  else {
    PetscCall(get_attributes(sw, f));
    for (const P& p : arrivals) {
      const PetscInt c = loc.cell(p.r);
      const PetscInt i = c * b.cap + b.cnt[c]++;
      std::memcpy(&f.r[3 * i], p.r, 3 * sizeof(PetscReal));
      std::memcpy(&f.v[3 * i], p.v, 3 * sizeof(PetscReal));
    }
    PetscCall(restore_attributes(sw, f));
  }

  t.sort += omp_get_wtime() - t1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode fill_blocks(
  DM sw, const Local& loc, Blocks& b, const VecP& pts, PetscInt cap)
{
  PetscFunctionBeginUser;
  const PetscInt nc = loc.cells();
  b.cnt.assign(nc, 0);
  for (const P& p : pts)
    b.cnt[loc.cell(p.r)]++;
  for (PetscInt c : b.cnt)
    cap = std::max(cap, 2 * c);
  b.cap = cap;
  std::fill(b.cnt.begin(), b.cnt.end(), 0);

  PetscCall(DMSwarmSetLocalSizes(sw, nc * cap, 0));
  Attributes f;
  PetscCall(get_attributes(sw, f));
  for (const P& p : pts) {
    const PetscInt c = loc.cell(p.r);
    const PetscInt i = c * cap + b.cnt[c]++;
    std::memcpy(&f.r[3 * i], p.r, 3 * sizeof(PetscReal));
    std::memcpy(&f.v[3 * i], p.v, 3 * sizeof(PetscReal));
  }
  PetscCall(restore_attributes(sw, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode gather_blocks(DM sw, const Blocks& b, VecP& out)
{
  PetscFunctionBeginUser;
  Attributes f;
  PetscCall(get_attributes(sw, f));
  for (size_t c = 0; c < b.cnt.size(); c++) {
    for (PetscInt q = 0; q < b.cnt[c]; q++) {
      const PetscInt i = (PetscInt)c * b.cap + q;
      P p;
      std::memcpy(p.r, &f.r[3 * i], 3 * sizeof(PetscReal));
      std::memcpy(p.v, &f.v[3 * i], 3 * sizeof(PetscReal));
      out.push_back(p);
    }
  }
  PetscCall(restore_attributes(sw, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// Backend e: flat swarm fields without per-cell order. Only the particles that
// leave the rank are processed (a surface-sized set): they are found in one
// parallel pass, removed by swapping with the last particle and replaced by
// the incoming ones appended at the end. The cell order of the arrays decays
// and is restored by the counting sort every `-resort` steps (0: never).
PetscErrorCode run_flat(const Config& cfg, const VecP& init)
{
  PetscFunctionBeginUser;
  Local loc;
  PetscCall(create_local(cfg, 0, &loc));
  const PetscReal L = cfg.n;
  const PetscInt n = cfg.n;
  const PetscInt total = (PetscInt)init.size();
  PetscInt expected;
  PetscCallMPI(MPI_Allreduce(&total, &expected, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

  DM sw;
  PetscCall(create_swarm(init, loc, &sw));
  std::vector<PetscInt> offsets;
  PetscCall(counting_sort(sw, loc, offsets));  // Initial order, not timed
  std::vector<PetscInt> perm;

  std::vector<PetscReal> rho((size_t)n * n * n);
  Phases t;
  for (PetscInt step = 0; step < cfg.steps; step++) {
    PetscReal t0 = omp_get_wtime();
    PetscCall(snapshot(sw));
    PetscReal t1 = omp_get_wtime();
    t.prepare += t1 - t0;

    Attributes f;
    PetscCall(get_attributes(sw, f));

#pragma omp parallel for
    for (PetscInt i = 0; i < f.np; i++) {
      push(&f.rp[3 * i], &f.vp[3 * i], &f.r[3 * i], &f.v[3 * i], L);
    }
    PetscReal t2 = omp_get_wtime();
    t.push += t2 - t1;

    std::fill(rho.begin(), rho.end(), 0);

#pragma omp parallel for
    for (PetscInt i = 0; i < f.np; i++) {
      deposit(rho.data(), n, &f.r[3 * i]);
    }
    t.deposit += omp_get_wtime() - t2;

    const PetscInt np = f.np;
    PetscCall(restore_attributes(sw, f));
    PetscCall(check_charge(rho, np));

    PetscCall(update_flat(sw, loc, L, t));
    PetscCall(check_count(sw, expected));

    if (resort_period > 0 && (step + 1) % resort_period == 0) {
      PetscReal t3 = omp_get_wtime();
      PetscCall(counting_sort(sw, loc, offsets));
      t.sort += omp_get_wtime() - t3;
    }

    // The cell order of `e` is stale: group the particles first, either by
    // a physical sort or by an index (data stay where they are)
    if (collide_period > 0 && (step + 1) % collide_period == 0) {
      PetscReal t3 = omp_get_wtime();
      if (collide_index)
        PetscCall(build_index(sw, loc, offsets, perm));
      else
        PetscCall(counting_sort(sw, loc, offsets));
      PetscCall(collide_swarm(sw, offsets, collide_index ? perm.data() : NULL, step));
      t.collide += omp_get_wtime() - t3;
    }
  }

  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&loc.da));
  PetscCall(report("swarm_e", t, cfg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode update_flat(DM sw, const Local& loc, PetscReal L, Phases& t)
{
  PetscFunctionBeginUser;
  PetscReal t0 = omp_get_wtime();

  const PetscMPIInt* neighbors;
  PetscCall(DMDAGetNeighbors(loc.da, &neighbors));

  Attributes f;
  PetscCall(get_attributes(sw, f));
  std::vector<std::vector<PetscInt>> movers(omp_get_max_threads());
#pragma omp parallel
  {
    std::vector<PetscInt>& mine = movers[omp_get_thread_num()];
#pragma omp for schedule(static)
    for (PetscInt i = 0; i < f.np; i++) {
      wrap(&f.r[3 * i], L);
      if (slot_of(loc, &f.r[3 * i]) != center)
        mine.push_back(i);
    }
  }

  // Threads own ascending chunks, so the indices are ascending in thread
  // order; removing from the largest one keeps the last element valid.
  VecP outgoing[slots];
  VecP incoming[slots];
  PetscInt np = f.np;
  for (auto vec = movers.rbegin(); vec != movers.rend(); vec++) {
    for (auto idx = vec->rbegin(); idx != vec->rend(); idx++) {
      P p;
      std::memcpy(p.r, &f.r[3 * *idx], 3 * sizeof(PetscReal));
      std::memcpy(p.v, &f.v[3 * *idx], 3 * sizeof(PetscReal));
      outgoing[slot_of(loc, p.r)].push_back(p);

      np--;
      if (*idx != np) {
        std::memcpy(&f.r[3 * *idx], &f.r[3 * np], 3 * sizeof(PetscReal));
        std::memcpy(&f.v[3 * *idx], &f.v[3 * np], 3 * sizeof(PetscReal));
      }
    }
  }
  PetscCall(restore_attributes(sw, f));

  PetscCall(exchange(neighbors, outgoing, incoming));

  PetscInt received = 0;
  for (const auto& in : incoming)
    received += (PetscInt)in.size();
  PetscCall(DMSwarmSetLocalSizes(sw, np + received, -1));

  PetscCall(get_attributes(sw, f));
  PetscInt w = np;
  for (const auto& in : incoming) {
    for (const P& p : in) {
      std::memcpy(&f.r[3 * w], p.r, 3 * sizeof(PetscReal));
      std::memcpy(&f.v[3 * w], p.v, 3 * sizeof(PetscReal));
      w++;
    }
  }
  PetscCall(restore_attributes(sw, f));

  t.migrate += omp_get_wtime() - t0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace particle_storage
