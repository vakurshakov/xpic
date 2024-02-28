#include "particles.h"

#include "src/impls/basic/simulation.h"

namespace basic {

Particles::Particles(const Simulation& simulation, const Particles_parameters& parameters)
  : simulation_(simulation), parameters_(parameters) {
  const DM& da = simulation_.da();

	PetscCallVoid(DMDAGetNeighbors(da, &neighbours));

  Vector3<PetscInt> start, size;
  PetscCallVoid(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

	l_start.x() = (PetscReal)start.x() * dx;
	l_start.y() = (PetscReal)start.y() * dy;
	l_start.z() = (PetscReal)start.z() * dz;

	l_end.x() = l_start.x() + (PetscReal)size.x() * dx;
	l_end.y() = l_start.y() + (PetscReal)size.y() * dy;
	l_end.z() = l_start.z() + (PetscReal)size.z() * dz;
}

PetscErrorCode Particles::add_particle(const Vector3<PetscReal>& r, const Vector3<PetscReal>& p) {
	PetscFunctionBeginUser;
  if (l_start.x() <= r.x() && r.x() < l_end.x() &&
      l_start.y() <= r.y() && r.y() < l_end.y() &&
      l_start.z() <= r.z() && r.z() < l_end.z()) {
    #pragma omp critical
    particles_.emplace_back(r, p, parameters_);
  }
	PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::push() {
	PetscFunctionBegin;
  #pragma omp for schedule(monotonic: dynamic, OMP_CHUNK_SIZE)
  for (auto it = particles_.begin(); it != particles_.end(); ++it) {
    // Vector3<PetscReal> r0 = it->r;
    Vector3<PetscReal> local_E = 0.0;
    Vector3<PetscReal> local_B = 0.0;

    push(*it, local_E, local_B);
  }
	PetscFunctionReturn(PETSC_SUCCESS);
}

void Particles::push(Particle& particle, const Vector3<PetscReal>& local_E, const Vector3<PetscReal>& local_B) const {
  PetscReal alpha = 0.5 * dt * particle.q();
  PetscReal m = particle.m();

  Vector3<PetscReal>& r = particle.r;
  Vector3<PetscReal>& p = particle.p;

  const Vector3<PetscReal> w = p + local_E * alpha;

  PetscReal energy = sqrt(m * m + w.dot(w));

  const Vector3<PetscReal> h = local_B * alpha / energy;

  const Vector3<PetscReal> s = h * 2.0 / (1.0 + h.dot(h));

  p = local_E * alpha + w * (1.0 - h.dot(s)) + w.cross(s) + h * (s.dot(w));

  energy = sqrt(m * m + p.dot(p));

  r += p * dt / energy;

  if (geom_nx == 1) r.x() = 0.5;
  if (geom_ny == 1) r.y() = 0.5;
  if (geom_nz == 1) r.z() = 0.5;
}

}
