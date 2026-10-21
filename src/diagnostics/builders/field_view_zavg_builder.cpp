#include "field_view_zavg_builder.h"

#include "src/diagnostics/field_view_zavg.h"

FieldViewZAvgBuilder::FieldViewZAvgBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : FieldViewBuilder(simulation, diagnostics)
{
}

PetscErrorCode FieldViewZAvgBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  std::string field;
  info.at("field").get_to(field);

  std::string out_dir = field + "_ZAvg";

  DM da;
  Vec f;
  Region region;
  parse_view(info, field, nullptr, da, f, region, out_dir);

  // The average is taken over the whole z-range, the output is 2D
  region.start[Z] = 0;
  region.size[Z] = 1;

  LOG("  Field view (z-average) diagnostic is added for {}, output directory: {}", field, out_dir);

  auto diagnostic = FieldViewZAvg::create(  //
    CONFIG().out_dir + "/" + out_dir, da, f, region);

  if (diagnostic)
    diagnostics_.emplace_back(std::move(diagnostic));
  PetscFunctionReturn(PETSC_SUCCESS);
}
