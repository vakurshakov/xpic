#ifndef SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
#define SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H

#include "src/interfaces/particles.h"
#include "src/diagnostics/field_view.h"


/**
 * @brief An utility structure to store the pointer to distribution moment
 * getter. Getter is chosen depending on the name in the constructor.
 */
struct Moment {
  const interfaces::Particles& particles_;

  using getter = PetscReal (*)(const interfaces::Particles&, const Point&);
  getter get = nullptr;

  Moment(const interfaces::Particles& particles, const getter& get);
  static Moment from_string(
    const interfaces::Particles& particles, const std::string& name);
};


/// @note The list of available moment getters
// clang-format off
inline PetscReal get_zeroth(const interfaces::Particles& particles, const Point& point);
inline PetscReal get_vx(const interfaces::Particles& particles, const Point& point);
inline PetscReal get_vy(const interfaces::Particles& particles, const Point& point);
inline PetscReal get_vz(const interfaces::Particles& particles, const Point& point);
inline PetscReal get_vr(const interfaces::Particles& particles, const Point& point);
inline PetscReal get_vphi(const interfaces::Particles& particles, const Point& point);
inline PetscReal get_m_vx_vx(const interfaces::Particles& particles, const Point& point);
inline PetscReal get_m_vx_vy(const interfaces::Particles& particles, const Point& point);
inline PetscReal get_m_vx_vz(const interfaces::Particles& particles, const Point& point);
inline PetscReal get_m_vy_vy(const interfaces::Particles& particles, const Point& point);
inline PetscReal get_m_vy_vz(const interfaces::Particles& particles, const Point& point);
inline PetscReal get_m_vz_vz(const interfaces::Particles& particles, const Point& point);
inline PetscReal get_m_vr_vr(const interfaces::Particles& particles, const Point& point);
inline PetscReal get_m_vr_vphi(const interfaces::Particles& particles, const Point& point);
inline PetscReal get_m_vphi_vphi(const interfaces::Particles& particles, const Point& point);
// clang-format on


/**
 * @brief Diagnostic of particles _coordinate_ distribution moment
 */
class DistributionMoment : public FieldView {
public:
  DEFAULT_MOVABLE(DistributionMoment);

  static std::unique_ptr<DistributionMoment> create(const std::string& out_dir,
    const interfaces::Particles& particles, const Moment& moment,
    const Region& region);

  ~DistributionMoment() override;

  PetscErrorCode diagnose(timestep_t t) override;

private:
  DistributionMoment(const std::string& out_dir,
    const interfaces::Particles& particles, const Moment& moment,
    MPI_Comm newcomm);

  PetscErrorCode set_data_views(const Region& region);
  PetscErrorCode set_da(const Region& region);

  PetscErrorCode collect();

  Vec local_;

  const interfaces::Particles& particles_;
  Moment moment_;
};

#endif  // SRC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
