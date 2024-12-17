#include "builder.h"

#include "src/utils/region_operations.h"


namespace interfaces {

Builder::Builder(const Simulation& simulation)
  : simulation_(simulation)
{
}

/* static */ Axis Builder::get_component(const std::string& name)
{
  if (name == "x")
    return X;
  if (name == "y")
    return Y;
  if (name == "z")
    return Z;
  throw std::runtime_error("Unknown component name " + name);
}

Vec Builder::get_global_vector(const std::string& name) const
{
  DM da = simulation_.world_.da;

  PetscBool flag;
  DMHasNamedGlobalVector(da, name.data(), &flag);
  if (!flag)
    throw std::runtime_error("Unknown field name " + name);

  Vec result;
  DMGetNamedGlobalVector(da, name.data(), &result);
  return result;
}


Vector3R Builder::parse_vector(
  const Configuration::json_t& json, const std::string& name) const
{
  std::string message;
  try {
    const Configuration::array_t& arr = json.at(name);

    if (arr.size() != 3) {
      message = name + " as array should be of size 3.";
      throw std::runtime_error(message);
    }

    Vector3R result;
    for (PetscInt i = 0; i < 3; ++i)
      arr[i].get_to(result[i]);
    return result;
  }
  catch (const std::exception& e) {
    message = e.what();
    message += usage_message();
    throw std::runtime_error(message);
  }
}


/* static */ PetscErrorCode Builder::check_region(
  const Vector3I& start, const Vector3I& size, const std::string& diag_name)
{
  PetscFunctionBeginUser;

  if (bool success = is_region_within_bounds(start, size, 0, Geom_n); !success) {
    throw std::runtime_error(
      "Region is not in global boundaries for " + diag_name + " diagnostic.");
  }

  if (bool success = (size[X] > 0) && (size[Y] > 0) && (size[Z] > 0); !success) {
    throw std::runtime_error(
      "Sizes are invalid for " + diag_name + " diagnostic.");
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace interfaces
