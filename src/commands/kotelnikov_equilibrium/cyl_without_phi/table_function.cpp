#include "table_function.h"

#include "src/utils/utils.h"

TableFunction::TableFunction(const std::string& filename)
{
  PetscCallAbort(PETSC_COMM_WORLD, evaluate_from_file(filename));
}

PetscErrorCode TableFunction::evaluate_from_file(const std::string& filename)
{
  PetscFunctionBeginUser;
  values_.clear();

  std::ifstream is(filename.data(), std::ios::binary);
  PetscCheck(is.good(), PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN,
    "Error occurred while reading table function from %s", filename.data());

  is.read((char*)&xmin_, sizeof(PetscReal));
  is.read((char*)&xmax_, sizeof(PetscReal));
  is.read((char*)&dx_, sizeof(PetscReal));

  std::vector<char> buf(std::istreambuf_iterator<char>(is), {});
  values_.reserve(buf.size() / sizeof(PetscReal));

  for (PetscSizeT i = 0; i < buf.size(); i += sizeof(PetscReal)) {
    auto value = *reinterpret_cast<PetscReal*>(buf.data() + i);
    values_.push_back(value);
  }

  PetscInt expected = ROUND_STEP(xmax_ - xmin_, dx_) + 1;
  PetscInt processed = (PetscInt)values_.size();

  PetscCheck(expected == processed, PETSC_COMM_WORLD, PETSC_ERR_FILE_READ,
    "Inconsistent data in the table from file %s, expected: %d, read: %d", filename.data(), expected, processed);
  PetscFunctionReturn(PETSC_SUCCESS);
}

void TableFunction::scale_coordinates(PetscReal scale)
{
  xmin_ *= scale;
  xmax_ *= scale;
  dx_ *= scale;
}

void TableFunction::scale_values(PetscReal scale)
{
  for (auto& v : values_)
    v *= scale;
}

PetscReal TableFunction::get_value(PetscReal x) const
{
  PetscFunctionBeginUser;
  PetscCheckAbort((xmin_ <= x && x <= xmax_), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Given x = %f [c/wpe] misses the boundaries of table function; xmin: %f, xmax: %f", x, xmin_, xmax_);

  x -= xmin_;
  PetscInt index = ROUND_STEP(x, dx_);

  if (dx_ < std::max({dx, dy, dz}))
    return values_[index];

  PetscReal result = 0;
  PetscReal delta_x = x / dx_ - index;

  if (0 <= index && index < (PetscInt)values_.size() / 2 - 1) {
    result = linear_interpolation(values_[index], values_[index + 1], delta_x);
  }
  else {
    // Take into account the last pair with no following values
    // Here `x` in (xmax_ - dx_/2, xmax_], so `delta_x` is negative
    delta_x = std::abs(delta_x);
    result = linear_interpolation(values_[index - 1], values_[index], delta_x);
  }
  PetscFunctionReturn(result);
}


PetscReal TableFunction::linear_interpolation(
  PetscReal v0, PetscReal v1, PetscReal t) const
{
  return (1.0 - t) * v0 + t * v1;
}
