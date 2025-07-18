#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: Magnetic mirror (probkotron). Particle should be reflected at mirror points, center stays between plugs.\n";

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal L = 10.0;

PetscReal get_B_mirror(PetscReal z) {
    return B_min + (B_max - B_min) * (z*z) / (L*L);
}

PetscReal get_gradB_mirror(PetscReal z) {
    return 2 * (B_max - B_min) * z / (L*L);
}

void get_mirror_field(const Vector3R& pos, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
    E_p = {0.0, 0.0, 0.0};
    PetscReal Bval = get_B_mirror(pos.z());
    PetscReal dBdz = get_gradB_mirror(pos.z());
    B_p = {0.0, 0.0, Bval};
    gradB_p = {0.0, 0.0, dBdz};
}

int main(int argc, char** argv)
{
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

    // Стартуем в центре, вблизи минимума поля
    constexpr Vector3R r0(0.0, 0.0, 0.0);
    // Дадим неполную скорость вдоль поля и поперёк
    constexpr PetscReal v_perp = 1.0;
    constexpr PetscReal v_par = 1.73; // 1.74
    constexpr Vector3R v0( v_perp, 0.0, v_par ); // (vx, vy, vz)
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
    push.set_qm(1.0); // заряд
    push.set_mp(1.0); // масса
    push.set_fields_callback(get_mirror_field);

    // Вычисляем максимальное z, до которого частица может дойти:
    //PetscReal mu = point_n.mu_p;
    //PetscReal E_tot = v_perp*v_perp + v_par*v_par;
    //PetscReal sin2_theta = mu * B_min / (E_tot/2);
    PetscReal z_max = L;
    // Внимание: mu = p_perp^2/(2mB), E_tot = v^2

    for (PetscInt t = 0; t <= geom_nt; ++t) {
        const PointByField point_0 = point_n;
        push.process(dt, point_n, point_0);
        PetscCall(trace.diagnose(t));
        // Можно добавить контроль — что z не выходит за пределы зеркал
        PetscCheck(std::abs(point_n.r.z()) <= z_max + 1e-2, PETSC_COMM_WORLD, PETSC_ERR_USER,
            "Particle must not escape magnetic mirror! z = %.6e, allowed max = %.6e",
            point_n.r.z(), z_max);
    }


    PetscFinalize();
    return EXIT_SUCCESS;
}
