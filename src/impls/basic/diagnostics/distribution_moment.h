#ifndef SRC_BASIC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
#define SRC_BASIC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H

#include "src/impls/basic/diagnostics/field_view.h"
#include "src/impls/basic/particles.h"

namespace basic {

/**
 * @brief An utility structure to store the pointer to distribution moment getter.
 * Getter is chosen depending on the name in the constructor.
 */
struct Moment {
  const Particles& particles_;

  using getter = PetscReal(*)(const Particles&, const Point&);
  getter get = nullptr;

  Moment(const Particles& particles, const getter& get);
  static Moment from_string(const Particles& particles, const std::string& name);
};


/// @note The list of available moment getters
inline PetscReal get_zeroth(const Particles&, const Point&);
inline PetscReal get_Vx(const Particles&, const Point&);
inline PetscReal get_Vy(const Particles&, const Point&);
inline PetscReal get_Vz(const Particles&, const Point&);
inline PetscReal get_Vr(const Particles&, const Point&);
inline PetscReal get_Vphi(const Particles&, const Point&);
inline PetscReal get_mVxVx(const Particles&, const Point&);
inline PetscReal get_mVxVy(const Particles&, const Point&);
inline PetscReal get_mVxVz(const Particles&, const Point&);
inline PetscReal get_mVyVy(const Particles&, const Point&);
inline PetscReal get_mVyVz(const Particles&, const Point&);
inline PetscReal get_mVzVz(const Particles&, const Point&);
inline PetscReal get_mVrVr(const Particles&, const Point&);
inline PetscReal get_mVrVphi(const Particles&, const Point&);
inline PetscReal get_mVphiVphi(const Particles&, const Point&);


/**
 * @brief Diagnostic of particles _coordinate_ distribution moment
 */
class Distribution_moment : public Field_view {
public:
  static std::unique_ptr<Distribution_moment> create(const std::string& out_dir,
    DM da, const Particles& particles, const Moment& moment, const Region& region);

  ~Distribution_moment();

  PetscErrorCode diagnose(timestep_t t) override;

private:
  Distribution_moment(const std::string& out_dir, DM da,
    const Particles& particles, const Moment& moment, MPI_Comm newcomm);

  PetscErrorCode set_data_views(const Region& region);
  PetscErrorCode set_da(const Region& region);

  PetscErrorCode collect();

  Vec local_;

  const Particles& particles_;
  Moment moment_;
};

}

#endif  // SRC_BASIC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
