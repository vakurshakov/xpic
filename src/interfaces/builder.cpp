#include "builder.h"


namespace interfaces {

Builder::Builder(Simulation& simulation)
  : simulation_(simulation)
{
}

/* static */ Axis Builder::get_component(const std::string& name)
{
  if (name == "x" || name == "X")
    return X;
  if (name == "y" || name == "Y")
    return Y;
  if (name == "z" || name == "Z")
    return Z;
  throw std::runtime_error("Unknown component name " + name);
}

/* static */ Vector3R Builder::parse_vector(
  const Configuration::json_t& info, const std::string& name)
{
  const Configuration::json_t value = info.at(name);

  switch (value.type()) {
    case nlohmann::json::value_t::array: {
      const Configuration::array_t& arr = value;

      if ((PetscInt)arr.size() != 3)
        throw std::runtime_error(name + " vector should be of size 3.");

      Vector3R result;
      for (PetscInt i = 0; i < 3; ++i) {
        result[i] = parse_value(arr[i]);
      }
      return result;
    }
    case nlohmann::json::value_t::string: {
      auto str = value.get<std::string>();
      if (str == "Geom")
        return Vector3R{Geom};
      if (str == "Geom / 2")
        return Vector3R{geom_x / 2, geom_y / 2, geom_z / 2};
      break;
    }
    default:
      return parse_value(value);
  }
  return Vector3R{};
}

/* static */ PetscReal Builder::parse_value(const Configuration::json_t& value)
{
  if (value.type() != nlohmann::json::value_t::string)
    return value.get<PetscReal>();

  auto str = value.get<std::string>();

  if (str == "geom_x" || str == "geom_nx")
    return geom_x;
  if (str == "geom_y" || str == "geom_ny")
    return geom_y;
  if (str == "geom_z" || str == "geom_nz")
    return geom_z;

  if (str.ends_with(" [dx]"))
    return std::stod(str.substr(0, str.size() - 5)) * dx;
  if (str.ends_with(" [dy]"))
    return std::stod(str.substr(0, str.size() - 5)) * dy;
  if (str.ends_with(" [dz]"))
    return std::stod(str.substr(0, str.size() - 5)) * dz;
  if (str.ends_with(" [dt]"))
    return std::stod(str.substr(0, str.size() - 5)) * dt;

  if (str.ends_with(" [c/w_pe]") || str.ends_with(" [1/w_pe]"))
    return std::stod(str.substr(0, str.size() - 9));

  throw std::runtime_error("Unknown string format to convert: " + str);
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
