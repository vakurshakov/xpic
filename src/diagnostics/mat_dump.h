#ifndef SRC_DIAGNOSTICS_MAT_DUMP_H
#define SRC_DIAGNOSTICS_MAT_DUMP_H

#include <petscmat.h>
#include <petscviewer.h>

#include "src/pch.h"
#include "src/interfaces/diagnostic.h"

class MatDump : public interfaces::Diagnostic {
public:
  MatDump(const std::string& out_dir, Mat mat, const std::string& comp_dir = "");

  PetscErrorCode diagnose(timestep_t t) override;

private:
  PetscErrorCode compare(timestep_t t);

  Mat mat_;
  PetscViewer viewer_;
  std::string comp_dir_;
};

#endif  // SRC_DIAGNOSTICS_MAT_DUMP_H
