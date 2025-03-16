#include "diagnostic.h"

namespace interfaces {

Diagnostic::Diagnostic()
  : diagnose_period_(diagnose_period)
{
}

Diagnostic::Diagnostic(const std::string& out_dir)
  : out_dir_(out_dir), diagnose_period_(diagnose_period)
{
}

Diagnostic::Diagnostic(const std::string& out_dir, PetscInt diagnose_period)
  : out_dir_(out_dir), diagnose_period_(diagnose_period)
{
}


/* static */ std::string Diagnostic::format_time(PetscInt t)
{
  auto time_width = (PetscInt)std::to_string(geom_nt).size();
  std::stringstream ss;
  ss.width(time_width);
  ss.fill('0');
  ss << t;
  return ss.str();
}

}  // namespace interfaces
