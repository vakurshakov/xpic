#ifndef SRC_ALGORITHMS_CRANK_NICOLSON_PUSH_H
#define SRC_ALGORITHMS_CRANK_NICOLSON_PUSH_H

#include "src/interfaces/particles.h"
#include "src/interfaces/point.h"

class CrankNicolsonPush {
public:
  CrankNicolsonPush() = default;
  CrankNicolsonPush(PetscReal qm);

  /// @brief Sets numerical tolerances controlling iterative process.
  void set_tolerances(PetscReal atol, PetscReal rtol, PetscInt maxit);

  /// @brief Sets charge to mass ratio, assumed to be constant during the iteration.
  void set_qm(PetscReal qm);

  /// @brief Once `process()` is complete, this will return the number of nonlinear iterations used.
  PetscInt get_iteration_number() const;

  /// @brief Used during the internal iteration process to obtain
  /// fields acting on particle at time-centered coordinate.
  using SetFields = std::function<void(const Vector3R&, Vector3R&, Vector3R&)>;
  void set_fields_callback(SetFields&& callback);

  /// @brief Nonlinear move of point `pn` by timestep shift `dt`.
  /// @warning `pn` and `p0` cannot be the same as `pn` will be updated.
  void process(PetscReal dt, Point& pn, const Point& p0);

private:
  void update_v(PetscReal dt, Vector3R& vn, const Vector3R& v0) const;
  void update_r(PetscReal dt, Point& pn, const Point& p0) const;
  PetscReal get_residue(PetscReal dt, const Point& pn, const Point& p0) const;

  PetscInt it = 0;
  PetscInt maxit = 30;
  PetscReal atol = 1e-7;
  PetscReal rtol = 1e-7;

  PetscReal qm = 0;

  SetFields set_fields;
  Vector3R E_p;
  Vector3R B_p;
};

#endif  // SRC_ALGORITHMS_CRANK_NICOLSON_PUSH_H
