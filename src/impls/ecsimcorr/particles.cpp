#include "particles.h"

#include "src/algorithms/boris_push.h"
#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/impls/ecsimcorr/simulation.h"

namespace ecsimcorr {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world_, parameters), simulation_(simulation)
{
  PetscFunctionBeginUser;
  DM da = world_.da;
  PetscCallVoid(DMCreateLocalVector(da, &local_currI));
  PetscCallVoid(DMCreateLocalVector(da, &local_currJe));
  PetscCallVoid(DMCreateGlobalVector(da, &global_currI));
  PetscCallVoid(DMCreateGlobalVector(da, &global_currJe));

  PetscClassIdRegister("ecsimcorr::Particles", &classid);
  PetscLogEventRegister("first_push", classid, &events[0]);
  PetscLogEventRegister("fill_matL", classid, &events[1]);
  PetscLogEventRegister("second_push", classid, &events[2]);
  PetscLogEventRegister("final_update", classid, &events[3]);
  PetscFunctionReturnVoid();
}

Particles::~Particles()
{
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&local_currI));
  PetscCallVoid(VecDestroy(&local_currJe));
  PetscCallVoid(VecDestroy(&global_currI));
  PetscCallVoid(VecDestroy(&global_currJe));
  PetscFunctionReturnVoid();
}

PetscErrorCode Particles::init()
{
  PetscFunctionBeginUser;
  energy = 0.0;

#pragma omp parallel for reduction(+ : energy), \
  schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_)
    energy += 0.5 * (mass(point) / particles_number(point)) * point.p.squared();

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(local_currI, 0.0));
  PetscCall(VecSet(local_currJe, 0.0));
  PetscCall(VecSet(global_currI, 0.0));
  PetscCall(VecSet(global_currJe, 0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::first_push()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(world_.da, local_currJe, &currJe));

  PetscLogEventBegin(events[0], local_currJe, 0, 0, 0);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_) {
    point.old_r = point.r;

    BorisPush push;
    push.update_r((0.5 * dt), point, *this);

    Shape shape;
    shape.setup(point.old_r, point.r, shape_radius2, shape_func2);
    decompose_esirkepov_current(shape, point);
  }

  PetscLogEventEnd(events[0], local_currJe, 0, 0, 0);

  PetscCall(DMDAVecRestoreArrayWrite(world_.da, local_currJe, &currJe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::fill_lapenta_matrix(
  PetscInt* coo_i, PetscInt* coo_j, PetscReal* coo_v, bool& preallocated)
{
  PetscFunctionBeginUser;
  Vec local_B;
  Vector3R*** B;

  DM da = world_.da;
  PetscCall(DMGetLocalVector(da, &local_B));
  PetscCall(DMGlobalToLocal(da, simulation_.B, INSERT_VALUES, local_B));

  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayWrite(da, local_currI, &currI));

  PetscLogEventBegin(events[1], local_B, local_currI, 0, 0);

  // #pragma omp parallel
  for (PetscInt i = 0; i < (PetscInt)points_.size(); ++i) {
    auto& point = points_[i];

    Shape shape;
    shape.setup(point.r, shape_radius1, shape_func1);

    Vector3R B_p;
    SimpleInterpolation interpolation(shape);
    interpolation.process({}, {{B_p, B}});

    Vector3R b = 0.5 * dt * charge(point) / mass(point) * B_p;
    decompose_identity_current(shape, point, b);
    decompose_lapenta_matrix(shape, i, b, coo_i, coo_j, coo_v, preallocated);

    correct_coordinates(point);
  }

  PetscLogEventEnd(events[1], local_B, local_currI, 0, 0);

  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_currI, &currI));
  PetscCall(DMRestoreLocalVector(da, &local_B));

  PetscCall(DMLocalToGlobal(da, local_currI, ADD_VALUES, global_currI));
  PetscCall(VecAXPY(simulation_.currI, 1.0, global_currI));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::second_push()
{
  PetscFunctionBeginUser;
  Vec local_E;
  Vec local_B;
  Vector3R*** E;
  Vector3R*** B;

  DM da = world_.da;
  PetscCall(DMGetLocalVector(da, &local_E));
  PetscCall(DMGetLocalVector(da, &local_B));
  PetscCall(DMGlobalToLocal(da, simulation_.Ep, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, simulation_.B, INSERT_VALUES, local_B));

  PetscCall(DMDAVecGetArrayRead(da, local_E, &E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayWrite(da, local_currJe, &currJe));

  PetscLogEventBegin(events[2], 0, 0, 0, 0);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_) {
    const Vector3R old_r = point.r;

    Shape shape;
    shape.setup(point.r, shape_radius1, shape_func1);

    Vector3R E_p;
    Vector3R B_p;
    SimpleInterpolation interpolation(shape);
    interpolation.process({{E_p, E}}, {{B_p, B}});

    BorisPush push;
    push.update_fields(E_p, B_p);
    push.update_vEB(dt, point, *this);
    push.update_r((0.5 * dt), point, *this);

    shape.setup(old_r, point.r, shape_radius2, shape_func2);
    decompose_esirkepov_current(shape, point);

    correct_coordinates(point);
  }

  PetscLogEventEnd(events[2], 0, 0, 0, 0);

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_currJe, &currJe));

  PetscCall(DMLocalToGlobal(da, local_currJe, ADD_VALUES, global_currJe));
  PetscCall(VecAXPY(simulation_.currJe, 1.0, global_currJe));

  PetscCall(DMRestoreLocalVector(da, &local_E));
  PetscCall(DMRestoreLocalVector(da, &local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::final_update()
{
  PetscFunctionBeginUser;
  PetscCall(VecDot(global_currI, simulation_.Ep, &pred_w));
  PetscCall(VecDot(global_currJe, simulation_.Ec, &corr_w));

  PetscReal K0 = energy;
  PetscReal K = 0.0;
  PetscLogEventBegin(events[3], 0, 0, 0, 0);

#pragma omp parallel for reduction(+ : K), \
  schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_)
    K += 0.5 * (mass(point) / particles_number(point)) * point.p.squared();

  PetscReal lambda2 = 1.0 + dt * (corr_w - pred_w) / K;
  PetscReal lambda = std::sqrt(lambda2);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& point : points_)
    point.p *= lambda;

  PetscLogEventEnd(events[3], 0, 0, 0, 0);

  lambda_dK = (lambda2 - 1.0) * K;
  pred_dK = K - K0;
  corr_dK = lambda2 * K - K0;
  energy = lambda2 * K;

  LOG("  Velocity renormalization for \"{}\"", parameters_.sort_name);
  LOG("    predicted field work [(ECSIM) * E_pred]: {:.7f}", pred_w);
  LOG("    corrected field work [(Esirkepov) * E_corr]: {:.7f}", corr_w);
  LOG("    lambda: {:.7f}, lambda^2: {:.7f}", lambda, lambda2);
  LOG("    d(energy) pred.: {:.7f}, corr.: {:.7f}, lambda: {:.7f}", pred_dK, corr_dK, lambda_dK);
  LOG("    energy prev.: {:.7f}, curr.: {:.7f}, diff: {:.7f}", K0, energy, energy - K0 /* == corr_dK */);

  PetscFunctionReturn(PETSC_SUCCESS);
}


void Particles::decompose_esirkepov_current(const Shape& shape, const Point& point)
{
  PetscFunctionBeginHot;
  const PetscReal alpha =
    charge(point) * density(point) / (particles_number(point) * (6.0 * dt));

  EsirkepovDecomposition decomposition(shape, alpha);
  PetscCallVoid(decomposition.process(currJe));
}


void Particles::decompose_identity_current(
  const Shape& shape, const Point& point, const Vector3R& b)
{
  PetscFunctionBeginHot;
  const PetscReal betaI = density(point) / particles_number(point) *
    charge(point) / (1.0 + b.squared());

  const Vector3R& v = point.p;
  Vector3R I_p = betaI * (v + v.cross(b) + b * v.dot(b));

  SimpleDecomposition decomposition(shape, I_p);
  PetscCallVoid(decomposition.process(currI));
  PetscFunctionReturnVoid();
}


void Particles::decompose_lapenta_matrix(  //
  const Shape& shape, PetscInt i, const Vector3R& b, //
  PetscInt* coo_i, PetscInt* coo_j, PetscReal* coo_v, bool& preallocated)
{
  PetscFunctionBeginHot;
  const Point& point = points_[i];

  /// @todo Bad, because 1) This condition always fails;
  /// 2) There can be particles in the nearest cells;
  if ((Shape::make_r(point.old_r) - Shape::make_r(point.r)).abs_max() >= 0.5){
    LOG("  Preallocation of Lapenta matrix disabled by sort:{}, i:{}", parameters_.sort_name, i);
    preallocated = false;
  }

  const PetscReal betaL = density(point) / particles_number(point) *
    POW2(charge(point)) / (mass(point) * (1.0 + b.squared()));

  constexpr PetscReal shape_tolerance = 1e-10;

  /// @todo Remove logical duplication. How to extrapolate this result on all shapes?
  PetscReal no[3][2];
  PetscReal sh[3][2];

  PetscReal x = point.x() / dx;
  PetscReal y = point.y() / dy;
  PetscReal z = point.z() / dz;
  PetscReal x05 = x - 0.5;
  PetscReal y05 = y - 0.5;
  PetscReal z05 = z - 0.5;

  auto no_gx = (PetscInt)x;
  auto no_gy = (PetscInt)y;
  auto no_gz = (PetscInt)z;
  auto sh_gx = (PetscInt)x05;
  auto sh_gy = (PetscInt)y05;
  auto sh_gz = (PetscInt)z05;

  PetscInt i0 = shape.s_p(0, 0, 0);
  PetscInt i1 = shape.s_p(1, 1, 1);
  sh[X][0] = shape(i0, Sh, X);
  sh[Y][0] = shape(i0, Sh, Y);
  sh[Z][0] = shape(i0, Sh, Z);
  sh[X][1] = shape(i1, Sh, X);
  sh[Y][1] = shape(i1, Sh, Y);
  sh[Z][1] = shape(i1, Sh, Z);

  i0 = shape.s_p(no_gz - sh_gz + 0, no_gy - sh_gy + 0, no_gx - sh_gy + 0);
  i1 = shape.s_p(no_gz - sh_gz + 1, no_gy - sh_gy + 1, no_gx - sh_gy + 1);
  no[X][0] = shape(i0, No, X);
  no[Y][0] = shape(i0, No, Y);
  no[Z][0] = shape(i0, No, Z);
  no[X][1] = shape(i1, No, X);
  no[Y][1] = shape(i1, No, Y);
  no[Z][1] = shape(i1, No, Z);

  auto t_g = [](PetscInt zi, PetscInt yi, PetscInt xi,  //
               PetscInt zj, PetscInt yj, PetscInt xj) {
    return ((zi * 2 + yi) * 2 + xi) * 8 + ((zj * 2 + yj) * 2 + xj);
  };

  auto ind = [&](PetscInt g, PetscInt c1, PetscInt c2) {
    constexpr PetscInt shape_size = POW2(3 * POW3(2));
    return i * shape_size + g * 9 + (c1 * 3 + c2);
  };

  const PetscReal matB[3][3]{
    {1.0 + b[X] * b[X], +b[Z] + b[X] * b[Y], -b[Y] + b[X] * b[Z]},
    {-b[Z] + b[Y] * b[X], 1.0 + b[Y] * b[Y], +b[X] + b[Y] * b[Z]},
    {+b[Y] + b[Z] * b[X], -b[X] + b[Z] * b[Y], 1.0 + b[Z] * b[Z]},
  };

  // clang-format off
  for (PetscInt zi = 0; zi < 2; ++zi) {
  for (PetscInt yi = 0; yi < 2; ++yi) {
  for (PetscInt xi = 0; xi < 2; ++xi) {
    Vector3R si{
      no[Z][zi] * no[Y][yi] * sh[X][xi],
      no[Z][zi] * sh[Y][yi] * no[X][xi],
      sh[Z][zi] * no[Y][yi] * no[X][xi],
    };

    // Shifts of g'=g2 iteration
    for (PetscInt zj = 0; zj < 2; ++zj) {
    for (PetscInt yj = 0; yj < 2; ++yj) {
    for (PetscInt xj = 0; xj < 2; ++xj) {
      Vector3R sj{
        no[Z][zj] * no[Y][yj] * sh[X][xj],
        no[Z][zj] * sh[Y][yj] * no[X][xj],
        sh[Z][zj] * no[Y][yj] * no[X][xj],
      };

      if (si.abs_max() < shape_tolerance || sj.abs_max() < shape_tolerance)
        continue;

      PetscInt g = t_g(zi, yi, xi, zj, yj, xj);

      for (PetscInt c1 = 0; c1 < 3; ++c1) {
        coo_i[ind(g, X, c1)] = indexing::v_g(no_gz + zi, no_gy + yi, sh_gx + xi, X);
        coo_i[ind(g, Y, c1)] = indexing::v_g(no_gz + zi, sh_gy + yi, no_gx + xi, Y);
        coo_i[ind(g, Z, c1)] = indexing::v_g(sh_gz + zi, no_gy + yi, no_gx + xi, Z);

        coo_j[ind(g, c1, X)] = indexing::v_g(no_gz + zj, no_gy + yj, sh_gx + xj, X);
        coo_j[ind(g, c1, Y)] = indexing::v_g(no_gz + zj, sh_gy + yj, no_gx + xj, Y);
        coo_j[ind(g, c1, Z)] = indexing::v_g(sh_gz + zj, no_gy + yj, no_gx + xj, Z);

        for (PetscInt c2 = 0; c2 < 3; ++c2) {
          coo_v[ind(g, c1, c2)] = si[c1] * sj[c2] * betaL * matB[c1][c2];
        }
      }
    }}}
  }}}
  // clang-format on
  PetscFunctionReturnVoid();
}

}  // namespace ecsimcorr
