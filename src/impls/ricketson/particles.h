#ifndef SRC_RICKETSON_PARTICLES_H
#define SRC_RICKETSON_PARTICLES_H

#include <petscsnes.h>

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/utils/shape.h"
#include "src/utils/sync_binary_file.h"

namespace ricketson {

class Simulation;

class Particles : public interfaces::Particles {
public:
  Particles(const Particles& other) = delete;
  Particles& operator=(const Particles& other) = delete;

  Particles(Particles&& other) noexcept;
  Particles& operator=(Particles&& other) noexcept = delete;

  ~Particles() final;

  Particles(Simulation& simulation, const SortParameters& parameters);

  PetscErrorCode push();

private:
  struct Context {
    Vector3R ***E, ***B, ***DB;
    Vector3R E_p, B_p, DB_p;

    PetscReal dt, m, q;

    Shape shape;

    Vector3R x_n, v_n, x_h, v_h;
    Vector3R v_hp, v_ht, v_E, DB_pp, DB_pt;
    PetscReal v_En, u;

    PetscErrorCode update(const Vector3R& x_nn, const Vector3R& v_nn);
  };

  PetscErrorCode push(Point& point);
  PetscErrorCode adaptive_time_stepping(const Point& point);

  static PetscErrorCode form_picard_iteration(
    SNES snes, Vec vx, Vec vf, void* vctx);

  Simulation& simulation_;

  Vec local_E;
  Vec local_B;
  Vec local_DB;

  // Nonlinear solver environment
  const PetscInt solution_size = 6;  // (3 coordinates) + (3 velocities)
  SNES snes_;
  Vec solution_;
  Context ctx;

  SyncBinaryFile particle_iterations_log;
};

}  // namespace ricketson

#endif  // SRC_RICKETSON_PARTICLES_H
