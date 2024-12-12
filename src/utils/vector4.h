#ifndef SRC_UTILS_VECTOR4_H
#define SRC_UTILS_VECTOR4_H

#include <type_traits>

#include <petscsystypes.h>

#include "src/utils/utils.h"

template<typename T>
  requires std::is_arithmetic_v<T>
struct Vector4 {
  static constexpr PetscInt dim = 4;
  std::array<T, dim> data;

  constexpr Vector4()
    : data{
        static_cast<T>(0),
        static_cast<T>(0),
        static_cast<T>(0),
        static_cast<T>(0),
      }
  {
  }

  constexpr Vector4(const T& v)
    : data{v, v, v, v}
  {
  }

  constexpr Vector4(const T& x, const T& y, const T& z, const T& c)
    : data{x, y, z, c}
  {
  }

  constexpr Vector4(const T v[Vector4::dim])
    : data{v[X], v[Y], v[Z], v[C]}
  {
  }

  constexpr operator const T*() const
  {
    return data.data();
  }

  constexpr operator T*()
  {
    return data.data();
  }

  Vector4& operator+=(const Vector4& other)
  {
    data[X] += other[X];
    data[Y] += other[Y];
    data[Z] += other[Z];
    data[C] += other[C];
    return *this;
  }

  Vector4& operator-=(const Vector4& other)
  {
    data[X] -= other[X];
    data[Y] -= other[Y];
    data[Z] -= other[Z];
    data[C] -= other[C];
    return *this;
  }

  Vector4& operator*=(T scalar)
  {
    data[X] *= scalar;
    data[Y] *= scalar;
    data[Z] *= scalar;
    data[C] *= scalar;
    return *this;
  }


  Vector4 operator+(const Vector4& other) const
  {
    return Vector4{
      data[X] + other[X],
      data[Y] + other[Y],
      data[Z] + other[Z],
      data[C] + other[C],
    };
  }

  Vector4 operator-(const Vector4& other) const
  {
    return Vector4{
      data[X] - other[X],
      data[Y] - other[Y],
      data[Z] - other[Z],
      data[C] - other[C],
    };
  }

  Vector4 elementwise_product(const Vector4& other) const
  {
    return Vector4{
      data[X] * other[X],
      data[Y] * other[Y],
      data[Z] * other[Z],
      data[C] * other[C],
    };
  }

  Vector4<PetscReal> operator/(T scalar) const
    requires std::is_floating_point_v<T>
  {
    return Vector4{
      static_cast<PetscReal>(data[X]) / scalar,
      static_cast<PetscReal>(data[Y]) / scalar,
      static_cast<PetscReal>(data[Z]) / scalar,
      static_cast<PetscReal>(data[C]) / scalar,
    };
  }

  Vector4<PetscReal> normalized() const
    requires std::is_floating_point_v<T>
  {
    return operator/(length());
  }

  PetscReal length() const
    requires std::is_floating_point_v<T>
  {
    return sqrt(static_cast<PetscReal>(squared()));
  }

  T elements_product() const
  {
    return data[X] * data[Y] * data[Z] * data[C];
  }

  T dot(const Vector4& other) const
  {
    return                  //
      data[X] * other[X] +  //
      data[Y] * other[Y] +  //
      data[Z] * other[Z] +  //
      data[C] * other[C];
  }

  T squared() const
  {
    return dot(*this);
  }

  Vector4<PetscReal> parallel_to(const Vector4& ref) const
    requires std::is_floating_point_v<T>
  {
    return ((*this).dot(ref) * ref) / ref.squared();
  }

  Vector4<PetscReal> transverse_to(const Vector4& ref) const
    requires std::is_floating_point_v<T>
  {
    return (*this) - parallel_to(ref);
  }

  void swap_order()
  {
    std::swap(data[X], data[Z]);
  }

  // clang-format off: access specifiers
  T& x() { return data[X]; }
  T& y() { return data[Y]; }
  T& z() { return data[Z]; }
  T& c() { return data[C]; }
  const T& x() const { return data[X]; }
  const T& y() const { return data[Y]; }
  const T& z() const { return data[Z]; }
  const T& c() const { return data[C]; }
  // clang-format on

  friend std::ostream& operator<<(std::ostream& out, const Vector4& vector)
  {
    out << std::to_string(vector[X]) << " ";
    out << std::to_string(vector[Y]) << " ";
    out << std::to_string(vector[Z]) << " ";
    out << std::to_string(vector[C]) << " ";
    return out;
  }
};

using Vector4R = Vector4<PetscReal>;
using Vector4I = Vector4<PetscInt>;

template<typename T>
Vector4<T> operator*(const Vector4<T>& vector, T scalar)
{
  return Vector4{
    vector[X] * scalar,
    vector[Y] * scalar,
    vector[Z] * scalar,
    vector[C] * scalar,
  };
}

template<typename T>
Vector4<T> operator*(T scalar, const Vector4<T>& vector)
{
  return vector * scalar;
}

template<typename T>
Vector4<T> min(const Vector4<T>& lhs, const Vector4<T>& rhs)
{
  return Vector4{
    std::min(lhs[X], rhs[X]),
    std::min(lhs[Y], rhs[Y]),
    std::min(lhs[Z], rhs[Z]),
    std::min(lhs[C], rhs[C]),
  };
}

template<typename T>
Vector4<T> max(const Vector4<T>& lhs, const Vector4<T>& rhs)
{
  return Vector4{
    std::max(lhs[X], rhs[X]),
    std::max(lhs[Y], rhs[Y]),
    std::max(lhs[Z], rhs[Z]),
    std::max(lhs[C], rhs[C]),
  };
}

#endif  // SRC_UTILS_VECTOR4_H
