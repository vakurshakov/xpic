#include "particles.h"

#include "src/algorithms/crank_nicolson_push.h"
#include "src/algorithms/implicit_esirkepov.h"
#include "src/impls/eccapfim/simulation.h"

namespace eccapfim {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world, parameters),
    previous_storage(world.size.elements_product()),
    simulation_(simulation)
{
}

PetscReal Particles::get_average_iteration_number() const
{
  return avgit;
}

PetscReal Particles::get_average_number_of_traversed_cells() const
{
  return avgcell;
}

PetscInt Particles::get_maximum_number_of_traversed_cells() const
{
  return maxcell;
}


PetscErrorCode Particles::form_prediction(PetscReal**** I_arr, PetscReal**** L_arr)
{
  PetscFunctionBeginUser;
  PetscReal q = parameters.q;
  PetscReal m = parameters.m;
  PetscReal mpw = parameters.n / (PetscReal)parameters.Np;

  // #pragma omp parallel for
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    if (storage[g].empty())
      continue;

    PetscInt x, y, z;
    x = world.start[X] + g % world.size[X];
    y = world.start[Y] + (g / world.size[X]) % world.size[Y];
    z = world.start[Z] + (g / world.size[X]) / world.size[Y];

    Vector3R b = B_arr[z][y][x] * ((0.5 * dt) * q / m);
    PetscInt Ng = storage[g].size();
    PetscReal A_p = 0.25 * dt * dt * mpw * Ng * q * q / m / (1 + b.squared());

    PetscReal matB[3][3]{
      {1.0 + b[X] * b[X], +b[Z] + b[X] * b[Y], -b[Y] + b[X] * b[Z]},
      {-b[Z] + b[Y] * b[X], 1.0 + b[Y] * b[Y], +b[X] + b[Y] * b[Z]},
      {+b[Y] + b[Z] * b[X], -b[X] + b[Z] * b[Y], 1.0 + b[Z] * b[Z]},
    };

    PetscReal* I_v = I_arr[z][y][x];
    PetscReal* L_v = L_arr[z][y][x];

    PetscInt c1, c2, in, jn, kn, is, js, ks;
    PetscReal xn, yn, zn, xs, ys, zs;

    for (c1 = 0; c1 < 3; c1++) {
      for (c2 = 0; c2 < 3; c2++)
        L_v[c1 * 3 + c2] += A_p * matB[c1][c2];
    }

    for (const auto& [r, v] : storage[g]) {
      Vector3R I_p =
        q * mpw / (1. + b.squared()) * (v + v.cross(b) + v.dot(b) * b);

      xn = r[X] / dx;
      yn = r[Y] / dy;
      zn = r[Z] / dz;
      xs = xn - 0.5;
      ys = yn - 0.5;
      zs = zn - 0.5;

      in = (PetscInt)floor(xn);
      jn = (PetscInt)floor(yn);
      kn = (PetscInt)floor(zn);
      is = (PetscInt)floor(xs);
      js = (PetscInt)floor(ys);
      ks = (PetscInt)floor(zs);

      I_v[X] += I_p[X];
      I_v[Y] += I_p[Y];
      I_v[Z] += I_p[Z];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::form_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(da, J_loc, &J_arr));

  avgit = 0;
  avgcell = 0;
  maxcell = 0;

  PetscReal q = parameters.q;
  PetscReal m = parameters.m;
  PetscReal mpw = parameters.n / parameters.Np;

  PetscReal xb = (world.start[X] - 0.5) * dx;
  PetscReal yb = (world.start[Y] - 0.5) * dy;
  PetscReal zb = (world.start[Z] - 0.5) * dz;

  PetscReal xe = (world.end[X] + 0.5) * dx;
  PetscReal ye = (world.end[Y] + 0.5) * dy;
  PetscReal ze = (world.end[Z] + 0.5) * dz;

  PetscReal max = std::numeric_limits<double>::max();

  auto process_bound = [&](PetscReal vh, PetscReal x, PetscReal xb, PetscReal xe) {
    if (vh > 0 && abs(xe - x) > 1e-7)
      return (xe - x) / vh;
    else if (vh < 0 && abs(xb - x) > 1e-7)
      return (xb - x) / vh;
    else
      return max;
  };

  auto bound_periodic = [&](PetscReal& s, Axis axis) {
    if (s < 0.0) {
      s = Geom[axis] - (0.0 - s);
      return true;
    }
    else if (s > Geom[axis]) {
      s = 0.0 + (s - Geom[axis]);
      return true;
    }
    return false;
  };

  /// @todo(2) Measure the time to interpolate / push / decompose separately
#pragma omp parallel for reduction(+ : avgit, avgcell), reduction(max : maxcell)
  for (PetscInt g = 0; g < (PetscInt)storage.size(); ++g) {
    ImplicitEsirkepov util(E_arr, B_arr, J_arr);

    for (PetscInt i = 0; auto& curr : storage[g]) {
      curr = previous_storage[g][i];
      Point tmp = curr;

      PetscReal tau = 0, dtau = 0, dtx, dty, dtz;

      for (; tau < dt; tau += dtau) {
        auto& pn = curr;
        auto& p0 = tmp;

        // This is a guess, not a proper calculation of `dtau`
        Vector3R vh = 0.5 * (pn.p + p0.p);

        dtx = process_bound(vh.x(), p0.x(), xb, xe);
        dty = process_bound(vh.y(), p0.y(), yb, ye);
        dtz = process_bound(vh.z(), p0.z(), zb, ze);
        dtau = std::min({dt - tau, dtx, dty, dtz});

        const PetscReal a0 = q * mpw;
        const PetscReal alpha = 0.5 * dtau * (q / m);

        const PetscInt cn_maxit = 30;
        const PetscReal cn_atol = 0.5 * atol;
        const PetscReal cn_rtol = 0.5 * atol;

        PetscInt it = 0, s;
        PetscReal rn, r0, d, ds, bs;

        Vector3R E_p, B_p, rsn, rs0;
        std::vector<Vector3R> coords;

        auto set_fields = [&] {
          E_p = Vector3R{};
          B_p = Vector3R{};

          d = (pn.r - p0.r).length();
          coords = cell_traversal(pn.r, p0.r);

          for (s = 1; s < (PetscInt)coords.size(); s++) {
            rs0 = coords[s - 1];
            rsn = coords[s - 0];
            ds = (rsn - rs0).length();
            bs = (d > 0 ? ds / d : 1.0);

            // No (dtau / dt) here, this is a field on substep `tau`
            Vector3R Es_p, Bs_p;
            util.interpolate(Es_p, Bs_p, rsn, rs0);
            E_p += Es_p * bs;
            B_p += Bs_p * bs;
          }
        };

        auto get_residue = [&] {
          return ((pn.p - p0.p) - (dtau * q / m) * (E_p + vh.cross(B_p))).length();
        };

        set_fields();
        rn = r0 = get_residue();

        for (; rn > cn_atol + cn_rtol * r0 && it < cn_maxit; it++) {
          Vector3R a, b, w;
          a = alpha * E_p;
          b = alpha * B_p;
          w = p0.p + a;
          vh = (w + w.cross(b) + b * w.dot(b)) / (1.0 + b.squared());

          pn.r = p0.r + dtau * vh;
          pn.p = 2.0 * vh - p0.p;

          set_fields();
          rn = get_residue();
        }

        avgit += (PetscReal)it / size;
        avgcell += (PetscReal)(coords.size() - 1) / size;
        maxcell = std::max(maxcell, (PetscInt)(coords.size() - 1));

        d = (pn.r - p0.r).length();
        coords = cell_traversal(pn.r, p0.r);

        for (s = 1; s < (PetscInt)coords.size(); s++) {
          rs0 = coords[s - 1];
          rsn = coords[s - 0];
          ds = (rsn - rs0).length();
          bs = (d > 0 ? ds / d : 1.0);

          util.decompose(a0 * bs * (dtau / dt), vh, rsn, rs0);
        }

        bool reset = false;
        reset |= bound_periodic(pn.r[X], X);
        reset |= bound_periodic(pn.r[Y], Y);
        reset |= bound_periodic(pn.r[Z], Z);

        if (reset)
          p0 = pn;
      }

      i++;
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMLocalToGlobal(da, J_loc, ADD_VALUES, J));

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &avgit, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &avgcell, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &maxcell, 1, MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));

  PetscMPIInt size;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  avgit /= (PetscReal)size;
  avgcell /= (PetscReal)size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::prepare_storage()
{
  PetscFunctionBeginUser;
  size = 0;

  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    auto&& curr = storage[g];
    if (curr.empty())
      continue;

    auto&& prev = previous_storage[g];
    prev = std::vector(curr.begin(), curr.end());

    size += (PetscInt)curr.size();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


std::vector<Vector3R> cell_traversal(const Vector3R& end, const Vector3R& start)
{
  Vector3I curr{
    (PetscInt)std::round(start[X] / dx),
    (PetscInt)std::round(start[Y] / dy),
    (PetscInt)std::round(start[Z] / dz),
  };

  Vector3I last{
    (PetscInt)std::round(end[X] / dx),
    (PetscInt)std::round(end[Y] / dy),
    (PetscInt)std::round(end[Z] / dz),
  };

  if (curr == last) {
    return {start, end};
  }

  Vector3R dir = (end - start);
  PetscInt sx = dir[X] > 0 ? 1 : -1;
  PetscInt sy = dir[Y] > 0 ? 1 : -1;
  PetscInt sz = dir[Z] > 0 ? 1 : -1;

  Vector3R next{
    (curr[X] + sx * 0.5) * dx,
    (curr[Y] + sy * 0.5) * dy,
    (curr[Z] + sz * 0.5) * dz,
  };

  static const PetscReal max = std::numeric_limits<double>::max();

  PetscReal t;
  PetscReal tx = (dir[X] != 0) ? (next[X] - start[X]) / dir[X] : max;
  PetscReal ty = (dir[Y] != 0) ? (next[Y] - start[Y]) / dir[Y] : max;
  PetscReal tz = (dir[Z] != 0) ? (next[Z] - start[Z]) / dir[Z] : max;

  PetscReal dtx = (dir[X] != 0) ? dx / dir[X] * sx : 0.0;
  PetscReal dty = (dir[Y] != 0) ? dy / dir[Y] * sy : 0.0;
  PetscReal dtz = (dir[Z] != 0) ? dz / dir[Z] * sz : 0.0;

  std::vector<Vector3R> points;
  points.push_back(start);

  while (curr != last) {
    if (tx < ty) {
      if (tx < tz) {
        t = tx;
        curr[X] += sx;
        tx += dtx;
      }
      else {
        t = tz;
        curr[Z] += sz;
        tz += dtz;
      }
    }
    else {
      if (ty < tz) {
        t = ty;
        curr[Y] += sy;
        ty += dty;
      }
      else {
        t = tz;
        curr[Z] += sz;
        tz += dtz;
      }
    }

    points.push_back(start + dir * t);
  }

  points.push_back(end);
  return points;
}

}  // namespace eccapfim
