#ifndef SRC_VECTORS_VECTOR4_H
#define SRC_VECTORS_VECTOR4_H

#include "src/pch.h"
#include "src/utils/utils.h"

template<typename T>
struct Vector4 {
  static const PetscInt dim = 4;
  T data[dim];

  Vector4() : data{(T)0, (T)0, (T)0, (T)0} {}
  Vector4(const T& v) : data{v, v, v, v} {}
  Vector4(const T& x, const T& y, const T& z, const T& c) : data{x, y, z, c} {}
  Vector4(const T v[Vector4::dim]) : data{v[X], v[Y], v[Z], v[C]} {}

  operator const T*() const { return data; }
  operator T*() { return data; }

  Vector4& operator+=(const Vector4& other) {
    data[X] += other[X];
    data[Y] += other[Y];
    data[Z] += other[Z];
    data[C] += other[C];
    return *this;
  }

  Vector4& operator-=(const Vector4& other) {
    data[X] -= other[X];
    data[Y] -= other[Y];
    data[Z] -= other[Z];
    data[C] -= other[C];
    return *this;
  }

  Vector4& operator*=(const Vector4& other) {
    data[X] *= other[X];
    data[Y] *= other[Y];
    data[Z] *= other[Z];
    data[C] *= other[C];
    return *this;
  }

  Vector4& operator*=(T scalar) {
    data[X] *= scalar;
    data[Y] *= scalar;
    data[Z] *= scalar;
    data[C] *= scalar;
    return *this;
  }


  Vector4 operator+(const Vector4& other) const {
    return {
      data[X] + other[X],
      data[Y] + other[Y],
      data[Z] + other[Z],
      data[C] + other[C],
    };
  }

  Vector4 operator-(const Vector4& other) const {
    return {
      data[X] - other[X],
      data[Y] - other[Y],
      data[Z] - other[Z],
      data[C] - other[C],
    };
  }

  Vector4 operator*(const Vector4& other) const {
    return {
      data[X] * other[X],
      data[Y] * other[Y],
      data[Z] * other[Z],
      data[C] * other[C],
    };
  }

  Vector4<PetscReal> operator/(T scalar) const {
    return {
      (PetscReal)data[X] / scalar,
      (PetscReal)data[Y] / scalar,
      (PetscReal)data[Z] / scalar,
      (PetscReal)data[C] / scalar,
    };
  }

  Vector4<PetscReal> normalized() const {
    return operator/(length());
  }

  PetscReal length() const {
    return sqrt((PetscReal)square());
  }

  T dot(const Vector4& other) const {
    return
      data[X] * other[X] +
      data[Y] * other[Y] +
      data[Z] * other[Z];
      data[C] * other[C];
  }

  T square() const {
    return dot(*this);
  }

  Vector4 parallel_to(const Vector4& ref) const {
    return (*this).dot(ref) * ref;
  }

  Vector4 transverse_to(const Vector4& ref) const {
    return (*this) - parallel_to(ref);
  }

  void swap_order() {
    std::swap(data[X], data[Z]);
  }

  T& x() { return data[X]; }
  T& y() { return data[Y]; }
  T& z() { return data[Z]; }
  T& c() { return data[C]; }
  const T& x() const { return data[X]; }
  const T& y() const { return data[Y]; }
  const T& z() const { return data[Z]; }
  const T& c() const { return data[C]; }
};

using Vector4R = Vector4<PetscReal>;
using Vector4I = Vector4<PetscInt>;

template<typename T> Vector4<T> operator*(const Vector4<T>& vector, T scalar) {
  return {
    vector[X] * scalar,
    vector[Y] * scalar,
    vector[Z] * scalar,
    vector[C] * scalar,
  };
}

template<typename T> Vector4<T> operator*(T scalar, const Vector4<T>& vector) {
  return vector * scalar;
}

template<typename T> Vector4<T> min(const Vector4<T>& lhs, const Vector4<T>& rhs) {
  return {
    std::min(lhs[X], rhs[X]),
    std::min(lhs[Y], rhs[Y]),
    std::min(lhs[Z], rhs[Z]),
    std::min(lhs[C], rhs[C]),
  };
}

template<typename T> Vector4<T> max(const Vector4<T>& lhs, const Vector4<T>& rhs) {
  return {
    std::max(lhs[X], rhs[X]),
    std::max(lhs[Y], rhs[Y]),
    std::max(lhs[Z], rhs[Z]),
    std::max(lhs[C], rhs[C]),
  };
}

#endif  // SRC_VECTORS_VECTOR4_H
