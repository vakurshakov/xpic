#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: Magnetic mirror (probkotron) + constant azimuthal E_phi. "
  "Check that particle is pushed toward axis due to ExB drift.\n";

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal L = 10.0;
constexpr PetscReal Ephi0 = 0.3; // амплитуда азимутального электрического поля

PetscReal get_B_mirror(PetscReal z) {
    return B_min + (B_max - B_min) * (z*z) / (L*L);
}
PetscReal get_gradB_mirror(PetscReal z) {
    return 2 * (B_max - B_min) * z / (L*L);
}

void get_azimutal_E_field(const Vector3R& pos, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
    PetscReal x = pos.x();
    PetscReal y = pos.y();

    PetscReal Ex = Ephi0 * y ;
    PetscReal Ey = -Ephi0 * x;

    E_p = {Ex, Ey, 0.0};
    B_p = {0.0, 0.0, get_B_mirror(pos.z())};
    gradB_p = {0.0, 0.0, get_gradB_mirror(pos.z())};
}

int main(int argc, char** argv)
{
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

    constexpr PetscReal R0 = 0.8;
    constexpr Vector3R r0(R0, 0.0, 0.0);
    constexpr PetscReal v_perp = 1.0;
    constexpr PetscReal v_par = 1.0;
    constexpr Vector3R v0( v_perp, 0.0, v_par );
    Point point_init(r0, v0);

    PetscReal omega_dt;
    PetscCall(get_omega_dt(omega_dt));

    std::string id = std::format("omega_dt_{:.1f}_periodicE", omega_dt);

    dt = omega_dt / get_B_mirror(r0.z());
    geom_nt = 100'000;
    diagnose_period = geom_nt / 4;

    PointByField point_n(point_init, {0.0, 0.0, B_min}, 1.0);

    PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 123);

    DriftKineticPush push;
    push.set_qm(1.0);
    push.set_mp(1.0);
    push.set_fields_callback(get_azimutal_E_field);

    PetscReal r_sum = 0.;
    PetscReal r_min = 1e10, r_max = -1e10;

    for (PetscInt t = 0; t <= geom_nt; ++t) {
        const PointByField point_0 = point_n;

        PetscCall(trace.diagnose(t));
        push.process(dt, point_n, point_0);

        PetscReal r_now = std::sqrt(point_n.r.x() * point_n.r.x() + point_n.r.y() * point_n.r.y());
        r_sum += r_now / geom_nt;
        if (r_now < r_min) r_min = r_now;
        if (r_now > r_max) r_max = r_now;

        PetscCheck(std::abs(point_n.r.z()) <= L + 1e-2, PETSC_COMM_WORLD, PETSC_ERR_USER,
            "Particle must not escape magnetic mirror! z = %.6e, allowed max = %.6e",
            point_n.r.z(), L);

        // Проверка: r всегда меньше 1.0
        PetscCheck(r_now <= R0, PETSC_COMM_WORLD, PETSC_ERR_USER,
            "Particle went outside allowed radius! r = %.3e", r_now);     
    }
    std::cout << "Mean radius: " << r_sum << "   Min r: " << r_min << "   Max r: " << r_max << std::endl;

    // Проверка: частица должна приблизиться к оси, средний радиус должен быть меньше начального
    PetscCheck(r_sum < R0, PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Particle was not pushed to axis by E_phi: mean r = %.3e (start %.3e)", r_sum, R0);

    PetscFinalize();
    return EXIT_SUCCESS;
}
