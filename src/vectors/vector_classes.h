#ifndef SRC_VECTORS_VECTOR_CLASSES_H
#define SRC_VECTORS_VECTOR_CLASSES_H

#include "src/pch.h"
#include "src/utils/utils.h"

enum Axis : PetscInt {
  X = 0,
  Y = 1,
  Z = 2,
  C = 3,
};


#define VECTOR_DEFAULT_CONSTRUCTORS(N)               \
  constexpr Vector##N();                             \
  constexpr Vector##N(const T&);                     \
  constexpr Vector##N(REP##N##_N(const T& x));       \
  constexpr Vector##N(const T[Vector##N::dim]);      \

#define VECTOR_DEFAULT_OPERATIONS(N)                 \
  operator const T*() const { return data; }         \
  operator T*() { return data; }                     \
  \
  Vector##N& operator+=(const Vector##N& other);     \
  Vector##N& operator-=(const Vector##N& other);     \
  Vector##N& operator*=(const Vector##N& other);     \
  Vector##N& operator*=(T scalar);                   \
  \
  Vector##N operator+(const Vector##N& other) const; \
  Vector##N operator-(const Vector##N& other) const; \
  Vector##N operator*(const Vector##N& other) const; \
  \
  Vector##N<PetscReal> operator/(T scalar) const;    \
  Vector##N<PetscReal> normalized() const;           \
  PetscReal length() const;                          \
  \
  T dot(const Vector##N& other) const;               \
  T square() const;                                  \
  \
  void swap_order();                                 \

#define COMP1                                      \
  constexpr const T& x() const { return data[X]; } \
  constexpr T& x() { return data[X]; }             \

#define COMP2 COMP1                                \
  constexpr const T& y() const { return data[Y]; } \
  constexpr T& y() { return data[Y]; }             \

#define COMP3 COMP2                                \
  constexpr const T& z() const { return data[Z]; } \
  constexpr T& z() { return data[Z]; }             \

#define COMP4 COMP3                                \
  constexpr const T& c() const { return data[C]; } \
  constexpr T& c() { return data[C]; }             \

#define VECTOR_DEFAULT_ACCESS(N) COMP##N

#define VECTOR_COMMUTATIVE_MULTIPLICATION(N)                                         \
  template<typename T> Vector##N<T> operator*(const Vector##N<T>& vector, T scalar); \
  template<typename T> Vector##N<T> operator*(T scalar, const Vector##N<T>& vector); \

#define VECTOR_MINMAX_COMPARISON(N)                                                        \
  template<typename T> Vector##N<T> min(const Vector##N<T>& lhs, const Vector##N<T>& rhs); \
  template<typename T> Vector##N<T> max(const Vector##N<T>& lhs, const Vector##N<T>& rhs); \


template<typename T>
struct Vector3 {
  static const PetscInt dim = 3;
  T data[dim];

  VECTOR_DEFAULT_CONSTRUCTORS(3)
  VECTOR_DEFAULT_OPERATIONS(3)
  VECTOR_DEFAULT_ACCESS(3)

  Vector3 cross(const Vector3& other) const {
    return {
      + (data[Y] * other[Z] - data[Z] * other[Y]),
      - (data[X] * other[Z] - data[Z] * other[X]),
      + (data[X] * other[Y] - data[Y] * other[X]),
    };
  }
};
VECTOR_COMMUTATIVE_MULTIPLICATION(3)
VECTOR_MINMAX_COMPARISON(3)

using Vector3R = Vector3<PetscReal>;
using Vector3I = Vector3<PetscInt>;


template<typename T>
struct Vector4 {
  static const PetscInt dim = 4;
  T data[dim];

  VECTOR_DEFAULT_CONSTRUCTORS(4)
  VECTOR_DEFAULT_OPERATIONS(4)
  VECTOR_DEFAULT_ACCESS(4)
};
VECTOR_COMMUTATIVE_MULTIPLICATION(4)
VECTOR_MINMAX_COMPARISON(4)

using Vector4R = Vector4<PetscReal>;
using Vector4I = Vector4<PetscInt>;


#include "vector_classes.inl"

#endif  // SRC_VECTORS_VECTOR_CLASSES_H
