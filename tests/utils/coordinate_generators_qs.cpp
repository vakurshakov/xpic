#include "src/utils/particles_load.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing QS coordinate generators: sequence, copying, bounds and profiles.\n";

namespace {

constexpr PetscInt samples = 8192;
constexpr int bins = 8;
constexpr PetscReal tolerance = 0.002;

PetscErrorCode check_sequence()
{
  PetscFunctionBeginUser;
  HaltonGenerator sequence;
  for (const Vector3R& expected : {
    Vector3R{0.5, 1.0 / 3, 0.2},
    Vector3R{0.25, 2.0 / 3, 0.4},
    Vector3R{0.75, 1.0 / 9, 0.6},
  }) {
    PetscCheck(equal_tol(sequence(), expected, 1e-14), PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Incorrect Halton point");
  }

  CoordinateGenerator original = CoordinateInBoxQS{{{-2, 3, 7}, {4, 11, 17}}};
  original();
  CoordinateGenerator copy = original;
  PetscCheck(equal_tol(original(), copy(), 1e-14), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Copy must preserve the sequence index");

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_box(bool hump, Axis axis)
{
  PetscFunctionBeginUser;
  const BoxGeometry box{{-2, 3, 7}, {4, 11, 17}};
  CoordinateGenerator generate;
  if (hump)
    generate = CoordinateInBoxCosineHumpQS{box, axis};
  else
    generate = CoordinateInBoxQS{box};

  std::array<std::array<PetscInt, bins>, 3> histogram{};
  std::array<PetscInt, 64> joint{};
  for (PetscInt i = 0; i < samples; i++) {
    Vector3R coordinate = generate();
    int cell = 0;
    for (Axis a : {X, Y, Z}) {
      PetscReal u = (coordinate[a] - box.min[a]) / (box.max[a] - box.min[a]);

      PetscCheck(u >= 0 && u < 1, PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Coordinate outside the box");

      histogram[a][static_cast<int>(bins * u)]++;
      cell = 4 * cell + static_cast<int>(4 * u);
    }
    joint[cell]++;
  }

  for (Axis a : {X, Y, Z}) {
    for (int b = 0; b < bins; b++) {
      PetscReal lo = static_cast<PetscReal>(b) / bins;
      PetscReal hi = static_cast<PetscReal>(b + 1) / bins;
      PetscReal expected = hi - lo;
      if (hump && a == axis)
        expected -=
          (sin(2.0 * M_PI * hi) - sin(2.0 * M_PI * lo)) / (2.0 * M_PI);

      PetscReal observed = static_cast<PetscReal>(histogram[a][b]) / samples;
      PetscCheck(equal_tol(observed, expected, tolerance), PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Incorrect box profile: observed %g, expected %g", observed, expected);
    }
  }

  if (!hump) {
    for (PetscInt count : joint) {
      PetscCheck(equal_tol(static_cast<PetscReal>(count) / samples, 1.0 / 64, tolerance), PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Halton points must fill the three-dimensional box");
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_cylinder()
{
  PetscFunctionBeginUser;
  const CylinderGeometry cyl{{-2, 3, 7}, 4, 10};
  CoordinateInCylinderQS generate{cyl};
  std::array<std::array<PetscInt, bins>, 3> histogram{};
  for (PetscInt i = 0; i < samples; i++) {
    Vector3R r = generate() - cyl.center;
    PetscReal phi = atan2(r[Y], r[X]) / (2.0 * M_PI);
    if (phi < 0)
      phi++;
    Vector3R unit{
      (r[X] * r[X] + r[Y] * r[Y]) / (cyl.radius * cyl.radius),
      phi,
      r[Z] / cyl.height + 0.5,
    };

    for (Axis a : {X, Y, Z}) {
      PetscCheck(unit[a] >= 0 && unit[a] < 1, PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Coordinate outside the cylinder");

      histogram[a][static_cast<int>(bins * unit[a])]++;
    }
  }

  for (Axis a : {X, Y, Z}) {
    for (PetscInt count : histogram[a]) {
      PetscCheck(equal_tol(static_cast<PetscReal>(count) / samples, 1.0 / bins, tolerance), PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Cylinder loading must be uniform in volume");
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(check_sequence());
  PetscCall(check_box(false, Z));
  for (Axis a : {X, Y, Z})
    PetscCall(check_box(true, a));
  PetscCall(check_cylinder());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
