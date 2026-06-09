#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Test of single particle drift-kinetic motion in a magnetic mirror. \n"
  "Compact geometry (20x20x100) with homogeneous central field B=1 and mirrors B=2. \n";

void overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();
  Configuration::save(get_out_dir(__FILE__));

  drift_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());
  PetscCall(simulation.calculate());
  PetscCall(simulation.finalize());

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  // Компактная сетка для быстрого счета
  dx = 1.0;
  geom_nx = 10;
  geom_x = geom_nx * dx;
  geom_ny = 10;
  geom_y = geom_ny * dx;
  geom_nz = 200; // Ловушка длиной 100 ячеек
  geom_z = geom_nz * dx;

  // Временной шаг
  dt = 10.0; // Достаточно малый шаг для разрешения продольного движения
  geom_nt = 100000; // 2000 вычислительных единиц времени
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {
      "Geometry",
      {
        {"x", geom_x}, {"y", geom_y}, {"z", geom_z},
        {"t", geom_t},
        {"dx", dx}, {"dy", dx}, {"dz", dx}, {"dt", dt},
        {"diagnose_period", dt}, // Запись каждые 10 шагов
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", 1},
        {"n", 1}, // Ничтожно малая плотность для отключения self-force (токов)
        {"q", -1.0},
        {"m", +1.0},
        {"T", +0.0},
      }},
    },
    {
"Presets",
      {
        // Шаг 1: Заливаем все пространство однородным полем B0 = 1.0.
        // Использует VecStrideSet, поэтому перезапишет нули.
        // ВАЖНО: Мы НЕ используем "field_axpy": "B" на этом шаге!
        {
          {"command", "SetMagneticField"},
          {"field", "B0"},
          {
            "setter", { {"name", "SetUniformField"}, {"value", {0.0, 0.0, 0.2}} }
          },
        },
        // Шаг 2: Накладываем поле катушек поверх B0.
        // Использует +=, поэтому B0 станет равно (Uniform + Coils).
        // Добавляем "field_axpy": "B", чтобы итоговое поле скопировалось в рабочий вектор B.
        {
          {"command", "SetMagneticField"},
          {"field", "B0"},
          {"field_axpy", "B"},
          {
            "setter",
            {
              {"name", "SetCoilsField"},
              {"coils", {
                {
                  {"z0", 50.0},
                  {"R",  10.0},
                  {"I",  0.32}
                },
                {
                  {"z0", 150.0},
                  {"R",  10.0},
                  {"I",  0.32}
                }
              }}
            },
          },
        },
        {
          {"command", "SetParticles"},
          {"particles", "electrons"},
          // Используем PreciseCoordinate вместо CoordinateList
          {"coordinate", {
            {"name", "PreciseCoordinate"},
            {"value", {5., 6.0, 100.0}} // Смещение по X, Y и центр по Z
          }},
          // Используем PreciseMomentum вместо MomentumList
          {"momentum", {
            {"name", "PreciseMomentum"},
            {"value", {0.1, 0.0, 0.05}} // p_perp = 0.1, p_par = 0.05
          }},
        }
      },
    },
          {
          "Diagnostics",
          {
            {
              {"diagnostic", "FieldView"},
              {"field", "B"},
              {"region", {{"type", "2D"}, {"plane", "X"}}},
            },
          },
      },
  });
}
