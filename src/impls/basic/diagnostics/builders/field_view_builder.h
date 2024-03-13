#ifndef SRC_BASIC_DIAGNOSTICS_BUILDERS_FIELD_VIEW_BUILDER_H
#define SRC_BASIC_DIAGNOSTICS_BUILDERS_FIELD_VIEW_BUILDER_H

#include "src/impls/basic/diagnostics/builders/diagnostic_builder.h"
#include "src/impls/basic/diagnostics/field_view.h"

namespace basic {

class Field_view_builder : public Diagnostic_builder {
public:
  Field_view_builder(const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics);

  PetscErrorCode build(const Configuration::json_t& diag_info) override;

private:
  struct Field_description {
    std::string field_name;
    std::string component_name;
    Field_view::Region region;
    MPI_Comm comm;
  };

  using Fields_description = std::vector<Field_description>;
  Fields_description fields_desc;

  PetscErrorCode parse_field_info(const Configuration::json_t& json, Field_description& desc);
  PetscErrorCode check_field_description(const Field_description& desc);
  PetscErrorCode attach_field_description(const DM& da, Field_description&& desc);
};

}

#endif  // SRC_BASIC_DIAGNOSTICS_BUILDERS_FIELD_VIEW_BUILDER_H
