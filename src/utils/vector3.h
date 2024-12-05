#ifndef SRC_UTILS_VECTOR3_H
#define SRC_UTILS_VECTOR3_H

#include <type_traits>

#include <petscsystypes.h>

#include "src/utils/utils.h"

template<typename T, typename std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
struct Vector3 {
  static constexpr PetscInt dim = 3;
  std::array<T, dim> data;

  constexpr Vector3()
    : data{
        static_cast<T>(0), //
        static_cast<T>(0), //
        static_cast<T>(0)  //
      }
  {
  }

  constexpr Vector3(const T& v)
    : data{v, v, v}
  {
  }

  constexpr Vector3(const T& x, const T& y, const T& z)
    : data{x, y, z}
  {
  }

  constexpr Vector3(const T v[Vector3::dim])
    : data{v[X], v[Y], v[Z]}
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

  Vector3& operator+=(const Vector3& other)
  {
    data[X] += other[X];
    data[Y] += other[Y];
    data[Z] += other[Z];
    return *this;
  }

  Vector3& operator-=(const Vector3& other)
  {
    data[X] -= other[X];
    data[Y] -= other[Y];
    data[Z] -= other[Z];
    return *this;
  }

  Vector3& operator*=(T scalar)
  {
    data[X] *= scalar;
    data[Y] *= scalar;
    data[Z] *= scalar;
    return *this;
  }

  template<typename U = T,
    typename std::enable_if_t<std::is_floating_point_v<U>, bool> = true>
  Vector3& operator/=(T scalar)
  {
    data[X] /= scalar;
    data[Y] /= scalar;
    data[Z] /= scalar;
    return *this;
  }

  Vector3 operator+(const Vector3& other) const
  {
    return Vector3{
      data[X] + other[X],
      data[Y] + other[Y],
      data[Z] + other[Z],
    };
  }

  Vector3 operator-(const Vector3& other) const
  {
    return Vector3{
      data[X] - other[X],
      data[Y] - other[Y],
      data[Z] - other[Z],
    };
  }

  Vector3 elementwise_product(const Vector3& other) const
  {
    return Vector3{
      data[X] * other[X],
      data[Y] * other[Y],
      data[Z] * other[Z],
    };
  }

  template<typename U = T,
    typename std::enable_if_t<std::is_floating_point_v<U>, bool> = true>
  Vector3<PetscReal> operator/(T scalar) const
  {
    return Vector3{
      static_cast<PetscReal>(data[X]) / scalar,
      static_cast<PetscReal>(data[Y]) / scalar,
      static_cast<PetscReal>(data[Z]) / scalar,
    };
  }

  template<typename U = T,
    typename std::enable_if_t<std::is_floating_point_v<U>, bool> = true>
  Vector3<PetscReal> normalized() const
  {
    return operator/(length());
  }

  template<typename U = T,
    typename std::enable_if_t<std::is_floating_point_v<U>, bool> = true>
  PetscReal length() const
  {
    return sqrt(static_cast<PetscReal>(squared()));
  }

  T elements_product() const
  {
    return data[X] * data[Y] * data[Z];
  }

  T dot(const Vector3& other) const
  {
    return                  //
      data[X] * other[X] +  //
      data[Y] * other[Y] +  //
      data[Z] * other[Z];
  }

  T squared() const
  {
    return dot(*this);
  }

  T abs_max()
  {
    return std::max(std::abs(data[X]), //
      std::max(std::abs(data[Y]), std::abs(data[Z])));
  }

  template<typename U = T,
    typename std::enable_if_t<std::is_floating_point_v<U>, bool> = true>
  Vector3<PetscReal> parallel_to(const Vector3& ref) const
  {
    return ((*this).dot(ref) * ref) / ref.squared();
  }

  template<typename U = T,
    typename std::enable_if_t<std::is_floating_point_v<U>, bool> = true>
  Vector3<PetscReal> transverse_to(const Vector3& ref) const
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
  const T& x() const { return data[X]; }
  const T& y() const { return data[Y]; }
  const T& z() const { return data[Z]; }
  // clang-format on

  Vector3 cross(const Vector3& other) const
  {
    return Vector3{
      +(data[Y] * other[Z] - data[Z] * other[Y]),
      -(data[X] * other[Z] - data[Z] * other[X]),
      +(data[X] * other[Y] - data[Y] * other[X]),
    };
  }
};

using Vector3R = Vector3<PetscReal>;
using Vector3I = Vector3<PetscInt>;

template<typename T>
Vector3<T> operator*(const Vector3<T>& vector, T scalar)
{
  return Vector3{
    vector[X] * scalar,
    vector[Y] * scalar,
    vector[Z] * scalar,
  };
}

template<typename T>
Vector3<T> operator*(T scalar, const Vector3<T>& vector)
{
  return vector * scalar;
}

template<typename T>
Vector3<T> min(const Vector3<T>& lhs, const Vector3<T>& rhs)
{
  return Vector3{
    std::min(lhs[X], rhs[X]),
    std::min(lhs[Y], rhs[Y]),
    std::min(lhs[Z], rhs[Z]),
  };
}

template<typename T>
Vector3<T> max(const Vector3<T>& lhs, const Vector3<T>& rhs)
{
  return Vector3{
    std::max(lhs[X], rhs[X]),
    std::max(lhs[Y], rhs[Y]),
    std::max(lhs[Z], rhs[Z]),
  };
}

#endif  // SRC_UTILS_VECTOR3_H
