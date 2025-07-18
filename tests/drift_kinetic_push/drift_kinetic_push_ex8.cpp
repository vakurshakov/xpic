#include "drift_kinetic_push.h"

static constexpr char help[] =
    "Test: Magnetic mirror (probkotron) field with realistic analytic mirrors. "
    "Particle should remain trapped in the well; energy conserved.\n";

// === Probkotron parameters ===
constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal L = 10.0;    // Trap length
constexpr PetscReal Delta = 1.0; // Mirror width (gauss)
constexpr PetscReal a = L/2;       // Half the length of the trap
constexpr PetscReal D = Delta*Delta;

inline PetscReal expL(PetscReal z) { return std::exp(-((z + a) * (z + a)) / D); }
inline PetscReal expR(PetscReal z) { return std::exp(-((z - a) * (z - a)) / D); }

// Center field profile on the axis (double gauss)
PetscReal get_Bz(PetscReal z) {
    return B_min + (B_max - B_min) * (expL(z) + expR(z));
}

// First derivative of z
PetscReal get_dBz_dz(PetscReal z) {
    PetscReal C = (B_max - B_min);
    return C * (
        -2.0 * (z + a) / D * expL(z)
        -2.0 * (z - a) / D * expR(z)
    );
}

// Second derivative of z
PetscReal get_d2Bz_dz2(PetscReal z) {
    PetscReal C = (B_max - B_min);
    return C * (
        (-2.0 / D + 4.0 * (z + a) * (z + a) / (D * D)) * expL(z)
        + (-2.0 / D + 4.0 * (z - a) * (z - a) / (D * D)) * expR(z)
    );
}

// Bz field off axis
PetscReal get_Bz_corr(const Vector3R& pos) {
    return get_Bz(pos.z()) - 0.25 * (pos.x()*pos.x() + pos.y()*pos.y()) * get_d2Bz_dz2(pos.z());
}

// Third derivative on z (for exact gradient Bz)
PetscReal get_d3Bz_dz3(PetscReal z) {
    PetscReal C = (B_max - B_min);
    PetscReal term1 = (z + a);
    PetscReal term2 = (z - a);
    return C * (
        (12.0 * term1 / (D*D) - 8.0 * term1*term1*term1 / (D*D*D)) * expL(z)
      + (12.0 * term2 / (D*D) - 8.0 * term2*term2*term2 / (D*D*D)) * expR(z)
    );
}

// The main function is field and gradient setting
void get_probe_trap_field(const Vector3R& pos, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
    // No electric field
    E_p = {0.0, 0.0, 0.0};

    // Coordinates and radius
    const PetscReal x = pos.x();
    const PetscReal y = pos.y();
    const PetscReal z = pos.z();
    const PetscReal r2 = x*x + y*y;
    const PetscReal r = std::sqrt(r2);

    // --- 1. Calculation of the axial field and its derivatives ---
    const PetscReal Bz_on_axis = get_Bz(z);
    const PetscReal dBz_dz = get_dBz_dz(z);
    const PetscReal d2Bz_dz2 = get_d2Bz_dz2(z);
    const PetscReal d3Bz_dz3 = get_d3Bz_dz3(z);

    // --- 2. Computation of the magnetic field vector B_p in the paraxial approximation ---
    // Transverse components, follow from div(B) = 0
    const PetscReal Bx = -0.5 * x * dBz_dz;
    const PetscReal By = -0.5 * y * dBz_dz;

    // Longitudinal component with second-order correction by radius
    const PetscReal Bz_corr = Bz_on_axis - 0.25 * r2 * d2Bz_dz2;

    // Total field vector
    B_p = {Bx, By, Bz_corr};

    // --- 3. Calculation of the field modulus gradient |B| ≈ Bz_corr ---

    // Radial component of the gradient: d|B|/dr
    const PetscReal dBmod_dr = -0.5 * r * d2Bz_dz2;

    // Longitudinal component of the gradient: d|B|/dz
    const PetscReal dBmod_dz = dBz_dz - 0.25 * r2 * d3Bz_dz3;

    // Collect the gradient vector from its components
    if (r > 1e-12) {
        gradB_p = Vector3R{ (x/r) * dBmod_dr,  
                             (y/r) * dBmod_dr,  
                             dBmod_dz };        
    } else {
        gradB_p = Vector3R{0.0, 0.0, dBmod_dz};
    }
}


int main(int argc, char** argv)
{
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

    constexpr Vector3R r0(0.5, 0.0, 0.0);
    constexpr PetscReal v_perp = 1.0;
    constexpr PetscReal v_par = 0.6;
    constexpr Vector3R v0(v_perp, 0.0, v_par);
    Point point_init(r0, v0);

    PetscReal omega_dt;
    PetscCall(get_omega_dt(omega_dt));

    std::string id = std::format("omega_dt_{:.1f}", omega_dt);

    dt = omega_dt / get_Bz(r0.z());
    geom_nt = 300'000;
    diagnose_period = geom_nt / 2;

    PointByField point_n(point_init, {0.0, 0.0, get_Bz_corr(r0)}, 1.0);

    PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 123);

    DriftKineticPush push;
    push.set_qm(1.0);
    push.set_mp(1.0);
    push.set_fields_callback(get_probe_trap_field);

    PetscReal z_max = a;
    PetscReal r_max = a;

    const PetscReal old_E = point_n.p_parallel * point_n.p_parallel +
                            point_n.p_perp * point_n.p_perp;

    for (PetscInt t = 0; t <= geom_nt; ++t) {
        const PointByField point_0 = point_n;
        push.process(dt, point_n, point_0);
        PetscCall(trace.diagnose(t));
        PetscCheck(std::abs(point_n.r.z()) <= z_max + 1e-2, PETSC_COMM_WORLD, PETSC_ERR_USER,
            "Particle escaped mirror! z = %.6e, allowed = %.6e",
            point_n.r.z(), z_max);
        PetscCheck(point_n.r.length() <= r_max + 1e-2, PETSC_COMM_WORLD, PETSC_ERR_USER,
            "Particle escaped radial well! r = %.6e, allowed = %.6e",
            point_n.r.length(), r_max);
    }
    PetscReal new_E = point_n.p_parallel * point_n.p_parallel +
                      point_n.p_perp * point_n.p_perp;
    PetscCheck(equal_tol(new_E, old_E, 1e-6), PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Energy not conserved: new = %.6e, old = %.6e", new_E, old_E);

    PetscFinalize();
    return EXIT_SUCCESS;
}
