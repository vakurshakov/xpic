#ifndef SRC_VECTORS_VECTOR3_H
#define SRC_VECTORS_VECTOR3_H

#include "src/pch.h"
#include "src/utils/utils.h"

template<typename T>
struct Vector3 {
  static const PetscInt dim = 3;
  T data[dim];

  Vector3()
    : data{(T)0, (T)0, (T)0}
  {
  }

  Vector3(const T& v)
    : data{v, v, v}
  {
  }

  Vector3(const T& x, const T& y, const T& z)
    : data{x, y, z}
  {
  }

  Vector3(const T v[Vector3::dim])
    : data{v[X], v[Y], v[Z]}
  {
  }

  operator const T*() const
  {
    return data;
  }

  operator T*()
  {
    return data;
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

  Vector3& operator*=(const Vector3& other)
  {
    data[X] *= other[X];
    data[Y] *= other[Y];
    data[Z] *= other[Z];
    return *this;
  }

  Vector3& operator*=(T scalar)
  {
    data[X] *= scalar;
    data[Y] *= scalar;
    data[Z] *= scalar;
    return *this;
  }


  Vector3 operator+(const Vector3& other) const
  {
    return {
      data[X] + other[X],
      data[Y] + other[Y],
      data[Z] + other[Z],
    };
  }

  Vector3 operator-(const Vector3& other) const
  {
    return {
      data[X] - other[X],
      data[Y] - other[Y],
      data[Z] - other[Z],
    };
  }

  Vector3 operator*(const Vector3& other) const
  {
    return {
      data[X] * other[X],
      data[Y] * other[Y],
      data[Z] * other[Z],
    };
  }

  Vector3<PetscReal> operator/(T scalar) const
  {
    return {
      (PetscReal)data[X] / scalar,
      (PetscReal)data[Y] / scalar,
      (PetscReal)data[Z] / scalar,
    };
  }

  Vector3<PetscReal> normalized() const
  {
    return operator/(length());
  }

  PetscReal length() const
  {
    return sqrt((PetscReal)square());
  }

  T dot(const Vector3& other) const
  {
    return                  //
      data[X] * other[X] +  //
      data[Y] * other[Y] +  //
      data[Z] * other[Z];
  }

  T square() const
  {
    return dot(*this);
  }

  Vector3<PetscReal> parallel_to(const Vector3& ref) const
  {
    return ((*this).dot(ref) * ref) / ref.square();
  }

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
    return {
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
  return {
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
  return {
    std::min(lhs[X], rhs[X]),
    std::min(lhs[Y], rhs[Y]),
    std::min(lhs[Z], rhs[Z]),
  };
}

template<typename T>
Vector3<T> max(const Vector3<T>& lhs, const Vector3<T>& rhs)
{
  return {
    std::max(lhs[X], rhs[X]),
    std::max(lhs[Y], rhs[Y]),
    std::max(lhs[Z], rhs[Z]),
  };
}

#endif  // SRC_VECTORS_VECTOR3_H
