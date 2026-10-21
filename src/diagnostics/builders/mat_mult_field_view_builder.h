#ifndef SRC_DIAGNOSTICS_BUILDERS_MAT_MULT_FIELD_VIEW_BUILDER_H
#define SRC_DIAGNOSTICS_BUILDERS_MAT_MULT_FIELD_VIEW_BUILDER_H

#include "src/diagnostics/builders/field_view_builder.h"

class MatMultFieldViewBuilder : public FieldViewBuilder {
public:
  using FieldViewBuilder::FieldViewBuilder;

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the MatMultFieldView diagnostic description:\n"
      "{\n"
      "  \"diagnostic\": \"MatMultFieldView\", -- Name of the diagnostic.\n"
      "  \"operator\": \"rotB\", -- One of: rotE, rotB, matM, divE.\n"
      "  \"field\": \"E\", -- Field the operator is applied to, same names\n"
      "                     as in FieldView.\n"
      "  \"component\", \"start\", \"size\", \"type\", \"out_dir\" -- Optional,\n"
      "                     the same as in FieldView. Default out_dir is\n"
      "                     \"<operator>_<field>\".\n"
      "}";
    return help;
  }
};

#endif  // SRC_DIAGNOSTICS_BUILDERS_MAT_MULT_FIELD_VIEW_BUILDER_H
