#ifndef SRC_DIAGNOSTICS_BUILDERS_FIELD_VIEW_ZAVG_BUILDER_H
#define SRC_DIAGNOSTICS_BUILDERS_FIELD_VIEW_ZAVG_BUILDER_H

#include "src/diagnostics/builders/field_view_builder.h"

class FieldViewZAvgBuilder : public FieldViewBuilder {
public:
  FieldViewZAvgBuilder(interfaces::Simulation& simulation,
    std::vector<Diagnostic_up>& diagnostics);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the FieldViewZAvg diagnostic description:\n"
      "{\n"
      "  \"diagnostic\": \"FieldViewZAvg\", -- Name of the diagnostic, "
      "                                        constant.\n"
      "  \"field\": \"E\", -- Field name set by `PetscObjectSetName()`.\n"
      "  \"start\": [ox, oy, -1], -- Starting point of a diagnosed region, in\n"
      "                              global coordinates of c/w_pe units.\n"
      "                              Optional, zeros will be used if empty.\n"
      "  \"size\": [sx, sy, -1] -- Sizes of a diagnosed region along each\n"
      "                            direction, in global coordinates of\n"
      "                            c/w_pe units. Optional, \"Geometry\"\n"
      "                            settings will be used if empty.\n"
      "}";
    return help;
  }
};

#endif  // SRC_DIAGNOSTICS_BUILDERS_FIELD_VIEW_ZAVG_BUILDER_H
