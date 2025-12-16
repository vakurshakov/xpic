#include "implicit_esirkepov.h"

#include "src/algorithms/simple_interpolation.h"

ImplicitEsirkepov::ImplicitEsirkepov(
  Vector3R*** E_g, Vector3R*** B_g, Vector3R*** J_g)
  : E_g(E_g), B_g(B_g), J_g(J_g)
{
}

void ImplicitEsirkepov::Shape::setup(const Vector3R& rn, const Vector3R& r0)
{
  Vector3R prn = ::Shape::make_r(rn);
  Vector3R pr0 = ::Shape::make_r(r0);
  Vector3R prh = 0.5 * (prn + pr0);

  Vector3R gc{
    std::round(prh[X]),
    std::round(prh[Y]),
    std::round(prh[Z]),
  };

  start = {
    (PetscInt)gc[X] - 1,
    (PetscInt)gc[Y] - 1,
    (PetscInt)gc[Z] - 1,
  };

  Vector3R gv{
    gc[X] + 0.5,
    gc[Y] + 0.5,
    gc[Z] + 0.5,
  };

  PetscInt i, j, k, cx, cy, cz, m = 0;
  PetscReal shx, sny, s0y, snz, s0z;

  static constexpr PetscReal sixth = 1.0 / 6.0;

  for (cx = 0; cx < 3; cx++) {
    cy = (cx + 1) % 3;
    cz = (cx + 2) % 3;

    for (i = 0; i < 2; i++) {
      shx = sixth * sfunc_1(gv[cx] + (i - 1) - prh[cx]);

      for (j = 0; j < 3; j++) {
        sny = sfunc_2[j](gc[cy] + (j - 1) - prn[cy]);
        s0y = sfunc_2[j](gc[cy] + (j - 1) - pr0[cy]);

        for (k = 0; k < 3; k++) {
          snz = sfunc_2[k](gc[cz] + (k - 1) - prn[cz]);
          s0z = sfunc_2[k](gc[cz] + (k - 1) - pr0[cz]);

          cache[m++] = shx * (sny * (2 * snz + s0z) + s0y * (2 * s0z + snz));
        }
      }
    }
  }
}


void ImplicitEsirkepov::interpolate(
  Vector3R& E_p, Vector3R& B_p, const Vector3R& rn, const Vector3R& r0)
{
  sh_m.setup(0.5 * (rn + r0));

  SimpleInterpolation util(sh_m);
  util.process({}, {{B_p, B_g}});

  sh_e.setup(rn, r0);

  PetscInt gx, gy, gz, cx, cy, cz, i[3], m = 0;

  for (cx = 0; cx < 3; cx++) {
    cy = (cx + 1) % 3;
    cz = (cx + 2) % 3;

    for (i[cx] = 0; i[cx] < 2; i[cx]++) {
      for (i[cy] = 0; i[cy] < 3; i[cy]++) {
        for (i[cz] = 0; i[cz] < 3; i[cz]++) {
          gx = sh_e.start[X] + i[X];
          gy = sh_e.start[Y] + i[Y];
          gz = sh_e.start[Z] + i[Z];

          E_p[cx] += E_g[gz][gy][gx][cx] * sh_e.cache[m++];
        }
      }
    }
  }
}

void ImplicitEsirkepov::decompose(
  PetscReal alpha, const Vector3R& v, const Vector3R& rn, const Vector3R& r0)
{
  sh_e.setup(rn, r0);

  PetscInt gx, gy, gz, cx, cy, cz, i[3], m = 0;

  for (cx = 0; cx < 3; cx++) {
    cy = (cx + 1) % 3;
    cz = (cx + 2) % 3;

    for (i[cx] = 0; i[cx] < 2; i[cx]++) {
      for (i[cy] = 0; i[cy] < 3; i[cy]++) {
        for (i[cz] = 0; i[cz] < 3; i[cz]++) {
          gx = sh_e.start[X] + i[X];
          gy = sh_e.start[Y] + i[Y];
          gz = sh_e.start[Z] + i[Z];

#pragma omp atomic
          J_g[gz][gy][gx][cx] += alpha * v[cx] * sh_e.cache[m++];
        }
      }
    }
  }
}
