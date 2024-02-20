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


static constexpr PetscInt Vector2_dim = 2;
static constexpr PetscInt Vector3_dim = 3;
static constexpr PetscInt Vector4_dim = 4;

#define VECTOR_DEFAULT_CONSTRUCTORS(VEC_T, N) \
  constexpr VEC_T();                          \
  constexpr VEC_T(const T&);                  \
  constexpr VEC_T(REP##N##_N(const T& x));    \
  constexpr VEC_T(const T[VEC_T##_dim]);      \

#define VECTOR_DEFAULT_OPERATIONS(VEC_T)      \
  operator const T*() const { return data; }  \
  operator T*() { return data; }              \
  \
  VEC_T& operator+=(const VEC_T& other);      \
  VEC_T& operator-=(const VEC_T& other);      \
  VEC_T& operator*=(T scalar);                \
  \
  VEC_T operator+(const VEC_T& other) const;  \
  VEC_T operator-(const VEC_T& other) const;  \
  VEC_T operator+() const;                    \
  VEC_T operator-() const;                    \
  \
  VEC_T<PetscReal> operator/(T scalar) const; \
  VEC_T<PetscReal> normalized() const;        \
  PetscReal length() const;                   \
  \
  T dot(const VEC_T& other) const;            \
  T square() const;                           \
  \
  VEC_T to_petsc_order() const;               \
  void to_petsc_order();                      \

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

#define VECTOR_COMMUTATIVE_MULTIPLICATION(VEC_T)                             \
  template<typename T> VEC_T<T> operator*(const VEC_T<T>& vector, T scalar); \
  template<typename T> VEC_T<T> operator*(T scalar, const VEC_T<T>& vector); \


template<typename T>
struct Vector2 {
  static constexpr Vector2   null{0, 0};
  static constexpr Vector2 orth_x{1, 0};
  static constexpr Vector2 orth_y{0, 1};

  T data[Vector2_dim];

  VECTOR_DEFAULT_CONSTRUCTORS(Vector2, 2)
  VECTOR_DEFAULT_OPERATIONS(Vector2)
  VECTOR_DEFAULT_ACCESS(2)
};
VECTOR_COMMUTATIVE_MULTIPLICATION(Vector2)


template<typename T>
struct Vector3 {
  static constexpr Vector3   null{0, 0, 0};
  static constexpr Vector3 orth_x{1, 0, 0};
  static constexpr Vector3 orth_y{0, 1, 0};
  static constexpr Vector3 orth_z{0, 0, 1};

  T data[Vector3_dim];

  VECTOR_DEFAULT_CONSTRUCTORS(Vector3, 3)
  VECTOR_DEFAULT_OPERATIONS(Vector3)
  VECTOR_DEFAULT_ACCESS(3)

  Vector3 cross(const Vector3& other) const;
  Vector2<T> squeeze_along(Axis axis) const;
};
VECTOR_COMMUTATIVE_MULTIPLICATION(Vector3)


template<typename T>
struct Vector4 {
  static constexpr Vector4   null{0, 0, 0, 0};
  static constexpr Vector4 orth_x{1, 0, 0, 0};
  static constexpr Vector4 orth_y{0, 1, 0, 0};
  static constexpr Vector4 orth_z{0, 0, 1, 0};
  static constexpr Vector4 orth_c{0, 0, 0, 1};

  T data[Vector4_dim];

  VECTOR_DEFAULT_CONSTRUCTORS(Vector4, 4)
  VECTOR_DEFAULT_OPERATIONS(Vector4)
  VECTOR_DEFAULT_ACCESS(4)
};
VECTOR_COMMUTATIVE_MULTIPLICATION(Vector4)

#include "vector_classes.inl"

#endif  // SRC_VECTORS_VECTOR_CLASSES_H
