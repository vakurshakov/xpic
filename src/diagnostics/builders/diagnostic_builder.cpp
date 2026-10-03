#include "diagnostic_builder.h"

#include "src/diagnostics/builders/distribution_moment_builder.h"
#include "src/diagnostics/builders/field_view_builder.h"
#include "src/diagnostics/builders/field_view_zavg_builder.h"
#include "src/diagnostics/builders/log_view_builder.h"
#include "src/diagnostics/builders/simulation_backup_builder.h"
#include "src/diagnostics/builders/velocity_distribution_builder.h"
#include "src/impls/drift_kinetic/diagnostic.h"
#include "src/impls/drift_kinetic/particles.h"
#include "src/utils/geometries.h"

DiagnosticBuilder::DiagnosticBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : Builder(simulation), diagnostics_(diagnostics)
{
}


PetscErrorCode build_diagnostics(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& result)
{
  PetscFunctionBeginUser;
  using namespace interfaces;

  const Configuration::json_t& config = CONFIG().json;

  LOG("Building diagnostics");

  auto&& it = config.find("SimulationBackup");
  if (it != config.end() && !it->empty())
    PetscCall(Builder::use_impl<SimulationBackupDiagBuilder>(*it, simulation, result));

  it = config.find("Diagnostics");
  if (it == config.end() || it->empty())
    PetscFunctionReturn(PETSC_SUCCESS);

  for (auto&& info : *it) {
    if (!info.contains("diagnostic"))
      continue;

    std::string name;
    info.at("diagnostic").get_to(name);

    if (name == "FieldView") {
      PetscCall(Builder::use_impl<FieldViewBuilder>(info, simulation, result));
    }
    else if (name == "FieldViewZAvg") {
      PetscCall(Builder::use_impl<FieldViewZAvgBuilder>(info, simulation, result));
    }
    else if (name == "DistributionMoment") {
      PetscCall(Builder::use_impl<DistributionMomentBuilder>(info, simulation, result));
    }
    else if (name == "VelocityDistribution") {
      PetscCall(Builder::use_impl<VelocityDistributionBuilder>(info, simulation, result));
    }
    else if (name == "DkDistributionFunction") {
      std::string particles_name;
      info.at("particles").get_to(particles_name);
      auto* particles = dynamic_cast<drift_kinetic::Particles*>(
        &simulation.get_named_particles(particles_name));
      if (particles == nullptr)
        throw std::runtime_error(
          "DkDistributionFunction requires drift_kinetic particles");

      auto parse_axis = [](const Configuration::json_t& axis_info,
                          const std::string& axis_name) {
        drift_kinetic::DkPhaseAxis axis;
        axis.min = interfaces::Builder::parse_value(axis_info.at("min"));
        axis.max = interfaces::Builder::parse_value(axis_info.at("max"));
        axis.bins = axis_info.at("bins").get<PetscInt>();
        if (!(axis.max > axis.min) || axis.bins <= 0)
          throw std::runtime_error(
            "Invalid DkDistributionFunction axis " + axis_name);
        return axis;
      };

      const auto v_parallel =
        parse_axis(info.at("v_parallel"), "v_parallel");
      const auto mu_p = parse_axis(info.at("mu_p"), "mu_p");
      const PetscInt period = info.contains("diagnose_period")
        ? std::max<PetscInt>(1, ROUND_STEP(
            interfaces::Builder::parse_value(info.at("diagnose_period")), dt))
        : diagnose_period;
      const PetscInt max_frames = info.contains("max_frames")
        ? info.at("max_frames").get<PetscInt>() : -1;
      if (max_frames == 0 || max_frames < -1)
        throw std::runtime_error(
          "DkDistributionFunction max_frames must be positive or omitted");
      std::string out_dir = particles_name + "/distribution_function";
      if (info.contains("out_dir"))
        info.at("out_dir").get_to(out_dir);
      out_dir = CONFIG().out_dir + "/" + out_dir;

      result.emplace_back(
        drift_kinetic::DkDistributionFunction::create(
          out_dir, *particles, v_parallel, mu_p, period, max_frames));
      LOG("  DkDistributionFunction diagnostic is added for {}, output directory: {}",
        particles_name, out_dir);
    }
    else if (name == "LogView") {
      PetscCall(Builder::use_impl<LogViewBuilder>(info, simulation, result));
    }
    else {
      throw std::runtime_error("Unknown diagnostic name " + name);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
