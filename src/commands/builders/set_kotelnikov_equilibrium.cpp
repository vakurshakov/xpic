#include "set_kotelnikov_equilibrium.h"

#include "src/commands/fields_damping.h"
#include "src/commands/inject_particles.h"
#include "src/commands/remove_particles.h"
#include "src/commands/set_magnetic_field.h"
//
#include "src/commands/kotelnikov_equilibrium/cyl_without_phi/set_cyl_without_phi.h"

namespace kotelnikov_equilibrium {

SetPresets::SetPresets(
  interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : ::CommandBuilder(simulation, result)
{
}

PetscErrorCode SetPresets::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  auto&& ions = simulation_.get_named_particles("ions");
  auto&& electrons = simulation_.get_named_particles("electrons");

  PetscReal mi = ions.parameters.m;
  PetscReal vi = std::sqrt(ions.parameters.Tx / (mi * mec2));

  using namespace cyl_without_phi;

  std::string params_str;
  info.at("parameters").get_to(params_str);

  auto gen_r = LoadCoordinate(params_str);
  gen_r.scale_coordinates(std::sqrt(mi));

  PetscInt particles_num = gen_r.get_cells_number() * ions.parameters.Np;

  auto gen_pi = LoadMomentum(ions.parameters, true, params_str);
  gen_pi.scale_coordinates(std::sqrt(mi));
  gen_pi.scale_chi(std::pow(mi, 1.5) * vi);

  auto gen_pe = MaxwellianMomentum(electrons.parameters, true);

  commands_.emplace_back(std::make_unique<InjectParticles>(
    ions, electrons, 0, 1, particles_num, gen_r, gen_pi, gen_pe));
  LOG("  InjectParticles command is added for Kotelnikov equilibrium");

  auto B = simulation_.get_named_vector("B");
  auto B0 = simulation_.get_named_vector("B0");

  auto setter = SetEquilibriumField(params_str);
  setter.scale_coordinates(std::sqrt(mi));
  setter.scale_b(std::pow(mi, 0.5) * vi);

  commands_.emplace_back(std::make_unique<SetMagneticField>(B0, B, setter));
  LOG("  SetMagneticField command is added for Kotelnikov equilibrium");
  PetscFunctionReturn(PETSC_SUCCESS);
}


SetStepPresets::SetStepPresets(
  interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : ParticlesBuilder(simulation, result)
{
}

PetscErrorCode SetStepPresets::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  using namespace cyl_without_phi;

  PetscReal buffer;
  info.at("buffer").get_to(buffer);

  Vector3R center(Geom);
  PetscReal radius = 0.5 * std::min(geom_x, geom_y) - buffer;
  PetscReal height = geom_z;
  CylinderGeometry cyl(0.5 * center, radius, height);

  auto test = [&]() {
    return WithinCylinder(cyl);
  };

  auto&& ions = simulation_.get_named_particles("ions");
  auto&& electrons = simulation_.get_named_particles("electrons");

  commands_.emplace_back(std::make_unique<RemoveParticles>(ions, test()));
  commands_.emplace_back(std::make_unique<RemoveParticles>(electrons, test()));
  LOG("  RemoveParticles commands are added for Kotelnikov equilibrium");

  // auto E = simulation_.get_named_vector("E");
  // auto B = simulation_.get_named_vector("B");
  // auto B0 = simulation_.get_named_vector("B0");

  // PetscReal coeff;
  // info.at("damping_coefficient").get_to(coeff);

  // auto damp = DampForCylinder(cyl, coeff);

  // commands_.emplace_back(std::make_unique<FieldsDamping>(
  //   simulation_.world.da, E, B, B0, test(), std::move(damp)));
  LOG("  FieldsDamping command is added for Kotelnikov equilibrium");
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace kotelnikov_equilibrium
