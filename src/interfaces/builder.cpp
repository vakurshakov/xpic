#include "builder.h"

#include "src/utils/region_operations.h"


namespace interfaces {

Builder::Builder(const Simulation& simulation)
  : simulation_(const_cast<Simulation&>(simulation))
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

Vector3R Builder::parse_vector(
  const Configuration::json_t& info, const std::string& name) const
{
  std::string message;
  const Configuration::json_t value = info.at(name);

  switch (value.type()) {
    case nlohmann::json::value_t::array: {
      const Configuration::array_t& arr = value;

      if (arr.size() != 3) {
        message = name + " vector should be of size 3.";
        throw std::runtime_error(message);
      }

      Vector3R result;
      for (PetscInt i = 0; i < 3; ++i)
        arr[i].get_to(result[i]);

      return result;
    }
    case nlohmann::json::value_t::string: {
      if (value.get<std::string>() == "Geom")
        return Vector3R{Geom};
      if (value.get<std::string>() == "Geom / 2")
        return Vector3R{geom_x / 2, geom_y / 2, geom_z / 2};
      break;
    }
    case nlohmann::json::value_t::number_integer:
    case nlohmann::json::value_t::number_unsigned:
    case nlohmann::json::value_t::number_float:
    default:
      return Vector3R{value.get<PetscReal>()};
  }
  return Vector3R{};
}


void Builder::check_region(
  const Vector3I& start, const Vector3I& size, const std::string& name) const
{
  if (bool success = is_region_within_bounds(start, size, 0, Geom_n); !success) {
    throw std::runtime_error(
      "Region is not in global boundaries for " + name + " diagnostic.");
  }

  if (bool success = (size[X] > 0) && (size[Y] > 0) && (size[Z] > 0); !success)
    throw std::runtime_error("Sizes are invalid for " + name + " diagnostic.");
}

}  // namespace interfaces
