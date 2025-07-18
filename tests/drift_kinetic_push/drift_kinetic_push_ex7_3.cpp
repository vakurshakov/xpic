#include "drift_kinetic_push.h"

static constexpr char help[] =
    "Test: Magnetic mirror (probkotron) field with radial well. "
    "Particle should remain trapped in the well; energy conserved.\n";

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal L = 10.0;   // полудлина ловушки
constexpr PetscReal Rc = 20.0;  // ширина радиальной ямы
constexpr PetscReal phi0 = 2.0;
constexpr PetscReal Ephi0 = 0.3;

PetscReal get_Bz(PetscReal z) {
    return B_min + (B_max - B_min) * (z*z) / (L*L);
}
PetscReal get_Bmod(PetscReal r, PetscReal z) {
    return get_Bz(z) * (1.0 + 0.5 * (r*r) / (Rc*Rc));
}
// Электрическое поле (направлено к центру)
PetscReal get_Ez(PetscReal z) {
    return -phi0 * M_PI / L * std::sin(M_PI * z / L);
}

void get_probe_trap_field(const Vector3R& pos, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{

    PetscReal x = pos.x();
    PetscReal y = pos.y();
    PetscReal z = pos.z();
    PetscReal r = std::sqrt(x*x + y*y);
    PetscReal Ex = Ephi0 * y ;
    PetscReal Ey = -Ephi0 * x;

    E_p = {Ex, Ey, get_Ez(pos.z())};

    PetscReal Bz = get_Bz(z);
    PetscReal Bmod = get_Bmod(r, z);

    B_p = {0.0, 0.0, Bmod};

    PetscReal dBz_dz = 2.0 * (B_max - B_min) * z / (L*L);
    PetscReal dBmod_dz = dBz_dz * (1.0 + 0.5 * (r*r)/(Rc*Rc));
    PetscReal dBmod_dr = Bz * r / (Rc*Rc);

    gradB_p = (r > 1e-10)
        ? Vector3R{x/r * dBmod_dr, y/r * dBmod_dr, dBmod_dz}
        : Vector3R{0.0, 0.0, dBmod_dz};
}

int main(int argc, char** argv)
{
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

    constexpr Vector3R r0(0.5, 0.0, 0.0);
    constexpr PetscReal v_perp = 1.0;
    constexpr PetscReal v_par = 1.0;
    constexpr Vector3R v0(v_perp, 0.0, v_par);
    Point point_init(r0, v0);

    PetscReal omega_dt;
    PetscCall(get_omega_dt(omega_dt));

    std::string id = std::format("omega_dt_{:.1f}", omega_dt);

    dt = omega_dt / get_Bmod(r0.length(), r0.z());
    geom_nt = 1'000;
    diagnose_period = geom_nt / 4;

    PointByField point_n(point_init, {0.0, 0.0, get_Bmod(r0.length(), r0.z())}, 1.0);

    PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 123);

    DriftKineticPush push;
    push.set_qm(1.0);
    push.set_mp(1.0);
    push.set_fields_callback(get_probe_trap_field);

    PetscReal z_max = L;
    PetscReal r_max = Rc;

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

    PetscFinalize();
    return EXIT_SUCCESS;
}
