#ifndef SRC_BASIC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
#define SRC_BASIC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H

#include "src/impls/basic/diagnostics/field_view.h"
#include "src/impls/basic/particles.h"

namespace basic {

struct Moment;
using Moment_up = std::unique_ptr<Moment>;

class Distribution_moment : public Field_view {
public:
  static std::unique_ptr<Distribution_moment> create(const std::string& out_dir,
    DM da, const Particles& particles, Moment_up moment, const Region& region);

  ~Distribution_moment();

  PetscErrorCode diagnose(timestep_t t) override;

private:
  Distribution_moment(const std::string& out_dir, DM da,
    const Particles& particles, Moment_up moment, MPI_Comm newcomm);

  PetscErrorCode set_data_views(const Region& region);
  PetscErrorCode set_da(const Region& region);

  PetscErrorCode collect();

  Vec local_;

  const Particles& particles_;
  Moment_up moment_;
};


/**
 * @brief An utility structure to store the pointer to distribution moment getter.
 * Getter is chosen depending on the name in the constructor.
 */
struct Moment {
  Moment(const Particles& particles, const std::string& name);

  const Particles& particles_;

  using getter = PetscReal(*)(const Particles&, const Point&);
  getter get = nullptr;
};

}

#endif  // SRC_BASIC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
