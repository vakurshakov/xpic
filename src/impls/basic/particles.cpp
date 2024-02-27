#include "particles.h"

#include "src/impls/basic/simulation.h"

namespace basic {

Particles::Particles(const Simulation& simulation, const Particles_parameters& parameters)
  : simulation_(simulation), parameters_(parameters) {}

void Particles::add_particle(const Vector3<PetscReal>& r, const Vector3<PetscReal>& p) {
  const DM& da = simulation_.da();

  Vector3<PetscInt> start, end;
  PetscCallVoid(DMDAGetCorners(da, REP3_A(&start), REP3_A(&end)));
  end += start;

  if (start.x() * dx <= r.x() && r.x() < end.x() * dx &&
      start.y() * dy <= r.y() && r.y() < end.y() * dy &&
      start.z() * dz <= r.z() && r.z() < end.z() * dz) {
    #pragma omp critical
    particles_.emplace_back(r, p, parameters_);
  }
}

void Particles::push() {
  #pragma omp for schedule(monotonic: dynamic, OMP_CHUNK_SIZE)
  for (auto it = particles_.begin(); it != particles_.end(); ++it) {
    // Vector3<PetscReal> r0 = it->r;
    Vector3<PetscReal> local_E = Vector3<PetscReal>::null;
    Vector3<PetscReal> local_B = Vector3<PetscReal>::null;

    push(*it, local_E, local_B);
  }
}

void Particles::push(Particle& particle, const Vector3<PetscReal>& local_E, const Vector3<PetscReal>& local_B) const
{
  const double alpha = 0.5 * dt * particle.q();
  const double m = particle.m();
  Vector3<PetscReal>& r = particle.r;
  Vector3<PetscReal>& p = particle.p;

  const Vector3<PetscReal> w = p + local_E * alpha;

  double energy = sqrt(m * m + w.dot(w));

  const Vector3<PetscReal> h = local_B * alpha / energy;

  const Vector3<PetscReal> s = h * 2. / (1. + h.dot(h));

  p = local_E * alpha + w * (1. - h.dot(s)) + w.cross(s) + h * (s.dot(w));

  energy = sqrt(m * m + p.dot(p));

  r += p * dt / energy;

  if (geom_nx == 1) r.x() = 0.5;
  if (geom_ny == 1) r.y() = 0.5;
  if (geom_nz == 1) r.z() = 0.5;
}

}
