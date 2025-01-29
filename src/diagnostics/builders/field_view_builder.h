#ifndef SRC_BASIC_BUILDERS_FIELD_VIEW_BUILDER_H
#define SRC_BASIC_BUILDERS_FIELD_VIEW_BUILDER_H

#include "src/diagnostics/builders/diagnostic_builder.h"
#include "src/diagnostics/field_view.h"

class FieldViewBuilder : public DiagnosticBuilder {
public:
  FieldViewBuilder(const interfaces::Simulation& simulation,
    std::vector<Diagnostic_up>& diagnostics);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the FieldView diagnostic description:\n"
      "{\n"
      "  \"diagnostic\": \"FieldView\", -- Name of the diagnostic, constant."
      "  \"field\": \"E\", -- Field name set by `PetscObjectSetName()`.\n"
      "  \"comp\":  \"x\", -- Diagnosed field component. Optional, with empty\n"
      "                       value it will print three components at once.\n"
      "  \"start\": [ox, oy, oz], -- Starting point of a diagnosed region, in\n"
      "                              global coordinates of c/w_pe units.\n"
      "                              Optional, zeros will be used if empty.\n"
      "  \"size\": [sx, sy, sz] -- Sizes of a diagnosed region along each\n"
      "                            direction, in global coordinates of\n"
      "                            c/w_pe units. Optional, \"Geometry\"\n"
      "                            settings will be used if empty.\n"
      "}";
    return help;
  }

protected:
  void parse_region_start_size(const Configuration::json_t& info,
    FieldView::Region& region, const std::string& name);
};

#endif  // SRC_BASIC_BUILDERS_FIELD_VIEW_BUILDER_H
