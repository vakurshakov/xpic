#ifndef SRC_DIAGNOSTICS_BUILDERS_FIELD_VIEW_BUILDER_H
#define SRC_DIAGNOSTICS_BUILDERS_FIELD_VIEW_BUILDER_H

#include "src/diagnostics/builders/diagnostic_builder.h"
#include "src/diagnostics/field_view.h"

class FieldViewBuilder : public DiagnosticBuilder {
public:
  FieldViewBuilder(interfaces::Simulation& simulation,
    std::vector<Diagnostic_up>& diagnostics);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the FieldView diagnostic description:\n"
      "{\n"
      "  \"diagnostic\": \"FieldView\", -- Name of the diagnostic, constant.\n"
      "  \"field\": \"E\", -- Field name set by `PetscObjectSetName()`.\n"
      "  \"component\": \"z\", -- Optional x/y/z component of a vector field.\n"
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

  /**
   * @brief Common part of building: resolves `field`, the diagnosed region
   * (scalar if `result_da` is `da_rho`, otherwise it is the vector one) and
   * the optional "region", "component" and "out_dir" keys.
   * @param result_da DM of the diagnosed data, `nullptr`
   * if it is the one of the field itself.
   */
  void parse_view(const Configuration::json_t& info, const std::string& field,
    DM result_da, DM& da, Vec& f, Region& region, std::string& out_dir);

  void parse_field(const Configuration::json_t& info, DM& da, Vec& f,
    Region& region, const std::string& name);

  void parse_region(
    const Configuration::json_t& info, Region& region, const std::string& name);

  void parse_plane_position(
    const Configuration::json_t& info, std::string& plane, PetscReal& position);

  void check_region(const Region& region, const std::string& name) const;
};

#endif  // SRC_DIAGNOSTICS_BUILDERS_FIELD_VIEW_BUILDER_H
