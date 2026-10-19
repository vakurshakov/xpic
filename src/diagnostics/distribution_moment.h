#ifndef SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
#define SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H

#include "src/interfaces/particles.h"
#include "src/diagnostics/field_view.h"

/// @note The size of the returned `vector` should match the passed `region.dof`
using Moment = std::vector<PetscReal> (*)(
  const interfaces::Particles&, const Point&);

/// @note To see the list of available moment getters, check the implementation
Moment moment_from_string(const std::string& name);

/// @brief Projects `v` at the position `r` onto the (R, A, Z) basis around the
/// domain axis. Near the axis the Cartesian components are returned unchanged.
Vector3R to_cylindrical(const Vector3R& r, const Vector3R& v);


/// @brief Diagnostic of particles _coordinate_ distribution moment
class DistributionMoment : public FieldView {
  friend class ChargeConservation;

public:
  static std::unique_ptr<DistributionMoment> create(const std::string& out_dir,
    const interfaces::Particles& particles, const Moment& moment,
    const Region& region);

  PetscErrorCode finalize() override;
  PetscErrorCode diagnose(PetscInt t) override;

protected:
  DistributionMoment(const interfaces::Particles& particles);

  DistributionMoment(const std::string& out_dir,
    const interfaces::Particles& particles, const Moment& moment,
    MPI_Comm newcomm);

  PetscErrorCode set_data_views(const Region& reg) override;

  struct Shape {
    static constexpr PetscInt shr = 1;
    static constexpr PetscInt shw = 2;
    static constexpr PetscInt shm = shw * shw * shw;

    Vector3I start;
    PetscReal cache[shm];

    void setup(const Vector3R& r);
  };

  virtual PetscErrorCode collect();

  DM da_glob;
  Vec field_loc;

  const interfaces::Particles& particles;
  Moment moment;
};

#endif  // SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
