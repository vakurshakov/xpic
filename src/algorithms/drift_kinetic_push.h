#ifndef SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H
#define SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H

#include "src/interfaces/particles.h"
#include "src/interfaces/point.h"

class DriftKineticPush {
public:
    DriftKineticPush() = default;
    DriftKineticPush(PetscReal qm, PetscReal mp);

    /// @brief Sets numerical tolerances controlling iterative process.
    void set_tolerances(PetscReal eps, PetscReal delta, PetscInt maxit);

    /// @brief Sets charge to mass ratio, assumed to be constant during the iteration.
    void set_qm(PetscReal qm);

    /// @brief Sets mass, assumed to be constant during the iteration.
    void set_mp(PetscReal mp);

    /// @brief Once `process()` is complete, this will return the number of nonlinear iterations used.
    PetscInt get_iteration_number() const;

    using SetFields = std::function<void(const Vector3R&, Vector3R&, Vector3R&, Vector3R&)>;
    void set_fields_callback(SetFields&& callback);

    /// @brief Nonlinear move of point `pn` by timestep shift `dt`.
    /// @warning `pn` and `p0` cannot be the same as `pn` will be updated.
    void process(PetscReal dt, PointByField& pn, const PointByField& p0);

private:
    // NEW
    Vector3R get_Vd(const PointByField& p0, const Vector3R& h, const PetscReal& Vh , const PetscReal& Bh, const Vector3R& gradBh, const PetscReal& Eh) const;
    void update_r(PetscReal dt, PointByField& pn, const PointByField& p0, const Vector3R& Vh, const Vector3R& Vd) const;
    void update_v_perp(PointByField& pn, const PointByField& p0, const Vector3R& B0) const;
    void update_v_parallel(PetscReal dt, PointByField& pn, const PointByField& p0, const PetscReal& Vh, const Vector3R& h, const Vector3R& Vd, const Vector3R& B0, const Vector3R& Eh) const;
    PetscReal get_residue_r(PetscReal dt, const PointByField& pn, const PointByField& p0, const Vector3R& Vh, const Vector3R& Vd) const;
    PetscReal get_residue_v(PetscReal dt, const PointByField& pn, const PointByField& p0, const PetscReal& Vh, const Vector3R& h, const Vector3R& Vd, const Vector3R& B0,  const Vector3R& Eh) const;

    PetscInt it = 0;
    PetscInt maxit = 30;
    PetscReal eps = 1e-7;
    PetscReal delta = 1e-7;

    PetscReal qm = 0;
    PetscReal mp = 0;

    SetFields set_fields;
    Vector3R Ep;
    Vector3R Bp;
    Vector3R gradBp;
};

#endif  // SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H