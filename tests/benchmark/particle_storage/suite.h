#ifndef TESTS_BENCHMARK_PARTICLE_STORAGE_SUITE_H
#define TESTS_BENCHMARK_PARTICLE_STORAGE_SUITE_H

#include <vector>

#include <petsc.h>

// Micro-benchmark of particle storage backends, see README.md.
namespace particle_storage {

// Options
extern bool xpic_tags;          // -xpic_tags: same tags as update_cells_mpi()
extern PetscReal omega_dt;      // -omega: gyration angle per step around z
extern PetscReal vth;           // -vth: thermal velocity, cells per step
extern PetscInt resort_period;  // -resort: counting sort period of `e`, 0: never
extern PetscInt buffer_percent;  // -buffer: swarm capacity reserve, % of particles
extern PetscInt stencil_override;     // -stencil: width of the swarm cell DM
extern PetscInt capacity_factor_x10;  // -capfac: block capacity / ppc, times 10
extern PetscInt collide_period;  // -collide: collisions every k steps, 0: off
extern bool collide_index;  // -collide_index: `e` groups by an index, not by a sort

constexpr PetscReal E0 = 1e-5;  // Amplitude of E_x ~ sin(2 pi x / n)
constexpr int slots = 27;
constexpr int center = 13;

struct P {
  PetscReal r[3];
  PetscReal v[3];
};

using VecP = std::vector<P>;

struct Config {
  PetscInt n = 32;      // Cells per axis (global, periodic)
  PetscInt ppc = 16;    // Particles per cell
  PetscInt steps = 20;  // Timesteps
};

struct Local {
  DM da;
  PetscInt s[3];  // First local cell
  PetscInt m[3];  // Local cells per axis

  PetscInt cells() const;
  PetscInt cell(const PetscReal* r) const;
};

struct Phases {
  double prepare = 0;
  double push = 0;
  double deposit = 0;
  double migrate = 0;
  double sort = 0;
  double collide = 0;
};

// Setup
PetscErrorCode create_local(const Config& cfg, PetscInt sw, Local* loc);
VecP make_particles(const Config& cfg, const Local& loc);

// Kernels and checks shared by all backends
void wrap(PetscReal*, PetscReal);
void push(const PetscReal*, const PetscReal*, PetscReal*, PetscReal*, PetscReal);
void deposit(PetscReal*, PetscInt, const PetscReal*);
PetscErrorCode check_charge(const std::vector<PetscReal>&, PetscInt);

// Takizuka-Abe like binary collisions of equal-mass particles inside a cell.
// Pairs are random, the odd particle of a cell is left alone. Momentum and
// kinetic energy are conserved per pair; `Moments` is used to check that.
struct Moments {
  PetscReal p[3] = {};
  PetscReal e = 0;
  void add(const PetscReal* v);
};
PetscErrorCode check_collisions(const Moments&, const Moments&, PetscReal);
PetscErrorCode report(const char*, const Phases&, const Config&);

// Neighbour exchange shared by all backends
int slot_of(const Local& loc, const PetscReal* r);
PetscErrorCode exchange(          //
  const PetscMPIInt* neighbors,   //
  const VecP (&outgoing)[slots],  //
  /*  */ VecP (&incoming)[slots]  //
);

// Backend l: `vector<list<P>>`
PetscErrorCode run_list(const Config& cfg, const Local& loc, const VecP& init);

// Backend v: `vector<vector<P>>`
PetscErrorCode run_vectors(const Config& cfg, const Local& loc, const VecP& init);

// Backends a, b, c: swarm with a cell-sorted layout
PetscErrorCode run_swarm(const Config& cfg, const VecP& init, char mode);

struct Fields {
  PetscInt np;
  PetscReal *r, *v, *rp, *vp;
};

PetscErrorCode create_swarm(const VecP& init, const Local& loc, DM* sw);
PetscErrorCode diagnose_cells(DM sw, const Config& cfg);
PetscErrorCode snapshot(DM sw);
PetscErrorCode wrap_and_migrate(DM sw, PetscReal L, Phases& t);
PetscErrorCode migrate_own(DM sw, const Local& loc, PetscReal L, Phases& t);
PetscErrorCode check_count(DM sw, PetscInt expected);
PetscErrorCode counting_sort(DM sw, const Local&, std::vector<PetscInt>&);
PetscErrorCode build_index(DM sw, const Local&, std::vector<PetscInt>& offsets,
  std::vector<PetscInt>& perm);
PetscErrorCode collide_swarm(DM sw, const std::vector<PetscInt>& offsets,
  const PetscInt* perm, PetscInt step);
PetscErrorCode get_fields(DM sw, Fields& f);
PetscErrorCode restore_fields(DM sw, Fields& f);

// Backend d: fixed-capacity per-cell blocks in the swarm fields
struct Blocks {
  PetscInt cap = 0;
  std::vector<PetscInt> cnt;
  PetscInt rebuilds = 0;
};

PetscErrorCode run_blocks(const Config& cfg, const VecP& init);
PetscErrorCode update_blocks(DM sw, const Local&, Blocks&, PetscReal, Phases&);
PetscErrorCode fill_blocks(DM sw, const Local&, Blocks&, const VecP&, PetscInt);
PetscErrorCode gather_blocks(DM sw, const Blocks& b, VecP& out);

// Backend e: flat swarm fields without per-cell order
PetscErrorCode run_flat(const Config& cfg, const VecP& init);
PetscErrorCode update_flat(DM sw, const Local& loc, PetscReal L, Phases& t);

}  // namespace particle_storage

#endif  // TESTS_BENCHMARK_PARTICLE_STORAGE_SUITE_H
