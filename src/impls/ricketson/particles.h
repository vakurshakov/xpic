#ifndef SRC_RICKETSON_PARTICLES_H
#define SRC_RICKETSON_PARTICLES_H

#include "src/interfaces/particles/particles.h"

#include <petscsnes.h>

#include "src/pch.h"
#include "src/vectors/vector3.h"
#include "src/impls/particle_shape.h"

namespace ricketson {

class Simulation;

class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const Particles_parameters& parameters);
  ~Particles();

  PetscErrorCode add_particle(const Point& point);
  const std::vector<Point>& get_points() const { return points_; }

  PetscErrorCode push();

private:
  struct Context {
    Vector3R ***E, ***B, ***DB;
    Vector3R E_p, B_p, DB_p;

    Vector3I width;
    PetscReal dt, m, q;

    Node node;
    Shape shape[2];

    Vector3R x_n, v_n, x_h, v_h;
    Vector3R v_hp, v_ht, v_E, DB_pp, DB_pt;
    PetscReal v_En, u;

    PetscErrorCode update(const Vector3R& x_nn, const Vector3R& v_nn);
    const PetscReal update_tolerance = 1e-8;
  };

  PetscErrorCode push(Point& point);
  PetscErrorCode adaptive_time_stepping(const Point& point);

  static PetscErrorCode form_Picard_iteration(SNES snes, Vec vx, Vec vf, void* context);

  std::vector<Point> points_;
  Simulation& simulation_;

  Vec local_E;
  Vec local_B;
  Vec local_DB;

  // Nonlinear solver environment
  const PetscInt solution_size = 6;  // (3 coordinates) + (3 velocities)
  SNES snes_;
  Vec solution_;
  Context ctx;
};

}

#endif  // SRC_RICKETSON_PARTICLES_H
