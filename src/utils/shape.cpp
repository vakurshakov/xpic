#include "shape.h"

/* static */ Vector3R Shape::make_r(const Vector3R& r)
{
  return Vector3R{
    r.x() / dx,
    r.y() / dy,
    r.z() / dz,
  };
}

/* static */ Vector3I Shape::make_g(const Vector3R& p_r)
{
  return Vector3I{
    static_cast<PetscInt>(std::round(p_r[X]) - shape_radius),
    static_cast<PetscInt>(std::round(p_r[Y]) - shape_radius),
    static_cast<PetscInt>(std::round(p_r[Z]) - shape_radius),
  };
}

/* static */ Vector3I Shape::make_start(const Vector3R& p_r, PetscReal radius)
{
  return Vector3I{
    static_cast<PetscInt>(std::round(p_r[X] - radius)),
    static_cast<PetscInt>(std::round(p_r[Y] - radius)),
    static_cast<PetscInt>(std::round(p_r[Z] - radius)),
  };
}

/* static */ Vector3I Shape::make_end(const Vector3R& p_r, PetscReal radius)
{
  return Vector3I{
    static_cast<PetscInt>(std::floor(p_r[X] + radius)) + 1,
    static_cast<PetscInt>(std::floor(p_r[Y] + radius)) + 1,
    static_cast<PetscInt>(std::floor(p_r[Z] + radius)) + 1,
  };
}


void Shape::setup(
  const Vector3R& r, PetscReal radius, PetscReal (&sfunc)(PetscReal))
{
  const Vector3R p_r = Shape::make_r(r);

  start = Shape::make_start(p_r, radius);
  size = Shape::make_end(p_r, radius);
  size -= start;

  fill(p_r, p_r, No, Sh, sfunc);
}

void Shape::setup(const Vector3R& old_r, const Vector3R& new_r,
  PetscReal radius, PetscReal (&sfunc)(PetscReal))
{
  const Vector3R old_p_r = Shape::make_r(old_r);
  const Vector3R new_p_r = Shape::make_r(new_r);

  start = Shape::make_start(min(old_p_r, new_p_r), radius);
  size = Shape::make_end(max(old_p_r, new_p_r), radius);
  size -= start;

  fill(old_p_r, new_p_r, Old, New, sfunc);
}


void Shape::fill(const Vector3R& p_r1, const Vector3R& p_r2, ShapeType t1,
  ShapeType t2, PetscReal (&sfunc)(PetscReal))
{
#pragma omp simd
  for (PetscInt i = 0; i < size.elements_product(); ++i) {
    auto g_x = static_cast<PetscReal>(start[X] + i % size[X]);
    auto g_y = static_cast<PetscReal>(start[Y] + (i / size[X]) % size[Y]);
    auto g_z = static_cast<PetscReal>(start[Z] + (i / size[X]) / size[Y]);

    shape[i_p(i, t1, X)] = sfunc(p_r1[X] - g_x);
    shape[i_p(i, t1, Y)] = sfunc(p_r1[Y] - g_y);
    shape[i_p(i, t1, Z)] = sfunc(p_r1[Z] - g_z);

    if (t2 == ShapeType::Sh) {
      g_x += 0.5;
      g_y += 0.5;
      g_z += 0.5;
    }

    shape[i_p(i, t2, X)] = sfunc(p_r2[X] - g_x);
    shape[i_p(i, t2, Y)] = sfunc(p_r2[Y] - g_y);
    shape[i_p(i, t2, Z)] = sfunc(p_r2[Z] - g_z);
  }
}
