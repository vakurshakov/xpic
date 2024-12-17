#ifndef SRC_BASIC_BUILDERS_FIELD_VIEW_BUILDER_H
#define SRC_BASIC_BUILDERS_FIELD_VIEW_BUILDER_H

#include "src/diagnostics/field_view.h"
#include "src/impls/basic/builders/diagnostic_builder.h"

class FieldViewBuilder : public DiagnosticBuilder {
public:
  FieldViewBuilder(const interfaces::Simulation& simulation,
    std::vector<Diagnostic_up>& diagnostics);

  PetscErrorCode build(const Configuration::json_t& diag_info) override;

private:
  std::string_view usage_message() const override
  {
    // clang-format off
    return "\n"
      "Usage: The structure of the field_view diagnostic description\n"
      "{\n"
      "  \"field\": \"E\", -- Diagnosed field that is represented in the `Simulation` class. Values: E, B.\n"
      "  \"comp\":  \"x\", -- Diagnosed field component, bounded by `DMDAGetDof()`. Optional, or empty value\n"
      "                   can be used explicitly to print all three components at once. Values: x, y, z or \"\".\n"
      "  \"start\": [ox, oy, oz], -- Starting point of a diagnostic in _global_ coordinates of units c/w_pe.\n"
      "  \"size\":  [sx, sy, sz]  -- Sizes of a diagnosed region along each coordinate in _global_ coordinates of units c/w_pe.\n"
      "}";
    // clang-format on
  }

  struct FieldDescription {
    std::string field_name;
    std::string component_name;
    FieldView::Region region;
    MPI_Comm comm;
  };

  using Fields_description = std::vector<FieldDescription>;
  Fields_description fields_desc_;

  PetscErrorCode parse_field_info(
    const Configuration::json_t& json, FieldDescription& desc);
};

#endif  // SRC_BASIC_BUILDERS_FIELD_VIEW_BUILDER_H
