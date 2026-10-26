#include "mat_mult_field_view_builder.h"

#include "src/diagnostics/mat_mult_field_view.h"
#include "src/utils/configuration.h"

PetscErrorCode MatMultFieldViewBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  std::string op_name, field;
  info.at("operator").get_to(op_name);
  info.at("field").get_to(field);

  // Operator and the DM of its result.
  const std::map<std::string, std::pair<Mat, DM>> operators{
    {"rotE", {simulation_.rotE, simulation_.da}},
    {"rotB", {simulation_.rotB, simulation_.da}},
    {"matM", {simulation_.matM, simulation_.da}},
    {"divE", {simulation_.divE, simulation_.da_rho}},
  };

  if (!operators.contains(op_name))
    throw std::runtime_error("Incorrect operator " + op_name + " is used.");

  auto [op, result_da] = operators.at(op_name);

  std::string out_dir = op_name + "_" + field;

  DM da;
  Vec source;
  Region region;
  parse_view(info, field, result_da, da, source, region, out_dir);

  LOG("  Mat-mult field view diagnostic is added for {} * {}, output directory: {}", op_name, field, out_dir);

  auto diagnostic = MatMultFieldView::create(  //
    CONFIG().out_dir + "/" + out_dir, result_da, op, source, region);

  if (diagnostic)
    diagnostics_.emplace_back(std::move(diagnostic));
  PetscFunctionReturn(PETSC_SUCCESS);
}
