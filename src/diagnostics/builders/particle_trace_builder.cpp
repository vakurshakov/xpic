#include "particle_trace_builder.h"

#include "src/diagnostics/particle_trace.h"
#include "src/impls/drift_kinetic/particles.h"

ParticleTraceBuilder::ParticleTraceBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : DiagnosticBuilder(simulation, diagnostics)
{
}

template<class PointT, class Arrays>
static std::optional<PointT> first_point(const ParticleStorage& storage)
{
  ParticleStorage::Attributes f;
  PetscCallAbort(PETSC_COMM_WORLD, storage.get_attributes(f));
  std::optional<PointT> point;
  if (f.size > 0)
    point = Arrays(f).load(0);
  PetscCallAbort(PETSC_COMM_WORLD, storage.restore_attributes(f));
  return point;
}

PetscErrorCode ParticleTraceBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  std::string name;
  info.at("particles").get_to(name);

  PetscInt skip = 1;
  if (info.contains("skip"))
    info.at("skip").get_to(skip);

  PetscCheckAbort(skip > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
    "ParticleTrace skip must be positive, got %" PetscInt_FMT, skip);

  const std::string filename = CONFIG().out_dir + "/temporal/particle_trace.txt";
  interfaces::Particles& particles = simulation_.get_named_particles(name);

  if (auto* dk = dynamic_cast<drift_kinetic::Particles*>(&particles)) {
    diagnostics_.emplace_back(std::make_unique<ParticleTrace<PointByField>>(
      filename,
      [dk]
      {
        using namespace drift_kinetic;
        return first_point<PointByField, DkPointArrays>(dk->storage);
      },
      skip));
  }
  else {
    diagnostics_.emplace_back(std::make_unique<ParticleTrace<Point>>(
      filename,
      [&particles]
      {
        return first_point<Point, PointArrays>(particles.storage);
      },
      skip));
  }

  LOG("  ParticleTrace diagnostic is added for {}, output file: {}", name, filename);
  PetscFunctionReturn(PETSC_SUCCESS);
}
