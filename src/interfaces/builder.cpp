#include "builder.h"


namespace interfaces {

Builder::Builder(Simulation& simulation)
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

Vector3R Builder::parse_vector(const Configuration::json_t& info,
  const std::string& name, PetscInt dim) const
{
  const Configuration::json_t value = info.at(name);

  switch (value.type()) {
    case nlohmann::json::value_t::array: {
      const Configuration::array_t& arr = value;

      if ((PetscInt)arr.size() != dim) {
        throw std::runtime_error(
          name + " vector should be of size " + std::to_string(dim) + ".");
      }

      static const std::vector<char> geom_map{'x', 'y', 'z'};

      Vector3R result;
      for (PetscInt i = 0; i < dim; ++i)
        if (arr[i].type() == nlohmann::json::value_t::string &&
          arr[i].get<std::string>() == std::string("geom_") + geom_map.at(i)) {
          result[i] = Geom[i];
        }
        else
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

void Builder::load_geometry(const Configuration::json_t& info, BoxGeometry& box)
{
  Vector3R min{0.0};
  Vector3R max{Geom};

  if (info.contains("min"))
    min = parse_vector(info, "min");
  if (info.contains("max"))
    max = parse_vector(info, "max");

  box = BoxGeometry(min, max);
}

void Builder::load_geometry(
  const Configuration::json_t& info, CylinderGeometry& cyl)
{
  Vector3R center{0.5 * geom_x, 0.5 * geom_y, 0.5 * geom_z};
  PetscReal radius = 0.5 * std::min(geom_x, geom_y);
  PetscReal height = geom_z;

  if (info.contains("center"))
    center = parse_vector(info, "center");
  if (info.contains("radius"))
    info.at("radius").get_to(radius);
  if (info.contains("height"))
    info.at("height").get_to(height);

  cyl = CylinderGeometry(center, radius, height);
}

}  // namespace interfaces
