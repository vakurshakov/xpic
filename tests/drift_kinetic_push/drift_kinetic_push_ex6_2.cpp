#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: Magnetic mirror (probkotron). Particle should be reflected at mirror points, center stays between plugs.\n";

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal L = 10.0;
constexpr PetscReal phi0 = 2.0;
constexpr PetscReal q = 1.0;
constexpr PetscReal m = 1.0;

PetscReal get_B_mirror(PetscReal z) {
    return B_min + (B_max - B_min) * (z*z) / (L*L);
}

PetscReal get_gradB_mirror(PetscReal z) {
    return 2 * (B_max - B_min) * z / (L*L);
}

PetscReal get_phi(PetscReal z) {
    return phi0 * std::sin(0.5 * M_PI * z / L) * std::sin(0.5 * M_PI * z / L);
}

PetscReal get_Ez(PetscReal z) {
    return -phi0 * M_PI / L * std::sin(M_PI * z / L);
}

PetscReal get_vpar2_critical_with_potential(
    PetscReal v_perp0, PetscReal B_min, PetscReal B_max, PetscReal phi0, PetscReal q, PetscReal m)
{
    return v_perp0 * v_perp0 * (B_max / B_min - 1.0) + 2.0 * q * phi0 / m;
}

PetscReal get_vpar2_critical_no_potential(
    PetscReal v_perp0, PetscReal B_min, PetscReal B_max)
{
    return v_perp0 * v_perp0 * (B_max / B_min - 1.0);
}

void get_mirror_potential_field(const Vector3R& pos, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
    E_p = {0.0, 0.0, get_Ez(pos.z())};
    PetscReal Bval = get_B_mirror(pos.z());
    PetscReal dBdz = get_gradB_mirror(pos.z());
    B_p = {0.0, 0.0, Bval};
    gradB_p = {0.0, 0.0, dBdz};
}

int main(int argc, char** argv)
{
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

    constexpr Vector3R r0(0.0, 0.0, 0.0);
    constexpr PetscReal v_perp = 1.0;
    constexpr PetscReal v_par = 2.0;
    constexpr Vector3R v0( v_perp, 0.0, v_par );
    Point point_init(r0, v0);

    PetscReal omega_dt;
    PetscCall(get_omega_dt(omega_dt));

    std::string id = std::format("omega_dt_{:.1f}", omega_dt);

    dt = omega_dt / get_B_mirror(r0.z());
    geom_nt = 100'000;
    diagnose_period = geom_nt / 4;

    PointByField point_n(point_init, {0.0, 0.0, B_min}, 1.0);

    PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 123);

    DriftKineticPush push;
    push.set_qm(q/m); 
    push.set_mp(m);
    push.set_fields_callback(get_mirror_potential_field);

    PetscReal vpar2_crit_with_potential = get_vpar2_critical_with_potential(v_perp, B_min, B_max, phi0, q, m);
    PetscReal vpar2_crit_no_potential   = get_vpar2_critical_no_potential(v_perp, B_min, B_max);

    std::cout << "v_parallel_critical (с потенциалом) = " << std::sqrt(vpar2_crit_with_potential) << std::endl;
    std::cout << "v_parallel_critical (без потенциала) = " << std::sqrt(vpar2_crit_no_potential) << std::endl;

    PetscReal z_max = L;

    for (PetscInt t = 0; t <= geom_nt; ++t) {
        const PointByField point_0 = point_n;
        push.process(dt, point_n, point_0);
        PetscCall(trace.diagnose(t));
        PetscCheck(std::abs(point_n.r.z()) <= z_max + 1e-2, PETSC_COMM_WORLD, PETSC_ERR_USER,
            "Particle must not escape magnetic mirror! z = %.6e, allowed max = %.6e",
            point_n.r.z(), z_max);
    }
    std::cout << z_max << std::endl;


    PetscFinalize();
    return EXIT_SUCCESS;
}
