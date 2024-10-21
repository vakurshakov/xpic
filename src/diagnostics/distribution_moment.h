#ifndef SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
#define SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H

#include "src/diagnostics/field_view.h"
#include "src/interfaces/particles.h"


/**
 * @brief An utility structure to store the pointer to distribution moment getter.
 * Getter is chosen depending on the name in the constructor.
 */
struct Moment {
  const interfaces::Particles& particles_;

  using getter = PetscReal(*)(const interfaces::Particles&, const Point&);
  getter get = nullptr;

  Moment(const interfaces::Particles& particles, const getter& get);
  static Moment from_string(const interfaces::Particles& particles, const std::string& name);
};


/// @note The list of available moment getters
inline PetscReal get_zeroth(const interfaces::Particles&, const Point&);
inline PetscReal get_Vx(const interfaces::Particles&, const Point&);
inline PetscReal get_Vy(const interfaces::Particles&, const Point&);
inline PetscReal get_Vz(const interfaces::Particles&, const Point&);
inline PetscReal get_Vr(const interfaces::Particles&, const Point&);
inline PetscReal get_Vphi(const interfaces::Particles&, const Point&);
inline PetscReal get_mVxVx(const interfaces::Particles&, const Point&);
inline PetscReal get_mVxVy(const interfaces::Particles&, const Point&);
inline PetscReal get_mVxVz(const interfaces::Particles&, const Point&);
inline PetscReal get_mVyVy(const interfaces::Particles&, const Point&);
inline PetscReal get_mVyVz(const interfaces::Particles&, const Point&);
inline PetscReal get_mVzVz(const interfaces::Particles&, const Point&);
inline PetscReal get_mVrVr(const interfaces::Particles&, const Point&);
inline PetscReal get_mVrVphi(const interfaces::Particles&, const Point&);
inline PetscReal get_mVphiVphi(const interfaces::Particles&, const Point&);


/**
 * @brief Diagnostic of particles _coordinate_ distribution moment
 */
class Distribution_moment : public Field_view {
public:
  static std::unique_ptr<Distribution_moment> create(const std::string& out_dir,
    const interfaces::Particles& particles, const Moment& moment, const Region& region);

  ~Distribution_moment();

  PetscErrorCode diagnose(timestep_t t) override;

private:
  Distribution_moment(const std::string& out_dir,
    const interfaces::Particles& particles, const Moment& moment, MPI_Comm newcomm);

  PetscErrorCode set_data_views(const Region& region);
  PetscErrorCode set_da(const Region& region);

  PetscErrorCode collect();

  Vec local_;

  const interfaces::Particles& particles_;
  Moment moment_;
};

#endif  // SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
