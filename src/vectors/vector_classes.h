#ifndef SRC_VECTORS_VECTOR_CLASSES_H
#define SRC_VECTORS_VECTOR_CLASSES_H

#include "src/pch.h"

enum Axis : PetscInt {
  X = 0,
  Y = 1,
  Z = 2
};


static constexpr PetscInt Vector2_dim = 2;

template<typename T>
struct Vector2 {

  static constexpr Vector2 null{0, 0};
  static constexpr Vector2 orth_x{1, 0};
  static constexpr Vector2 orth_y{0, 1};

  T x;
  T y;

  constexpr Vector2()
    : x{0}, y{0} {}

  constexpr Vector2(const T& _value)
    : x{_value}, y{_value} {}

  constexpr Vector2(const T& _x, const T& _y)
    : x{_x}, y{_y} {}

  constexpr Vector2(T _vector[Vector2_dim])
    : x{_vector[X]}, y{_vector[Y]} {}

  Vector2& operator+=(const Vector2& other);
  Vector2& operator-=(const Vector2& other);
  Vector2& operator*=(PetscReal scalar);
  Vector2& operator/=(PetscReal scalar);

  Vector2 operator+(const Vector2& other) const;
  Vector2 operator-(const Vector2& other) const;
  Vector2 operator/(PetscReal scalar) const;
  Vector2 operator+() const;
  Vector2 operator-() const;

  Vector2& normalize();
  Vector2 normalized() const;
  T dot(const Vector2& other) const;
  T square() const;
  PetscReal length() const;
};

template<typename T>
Vector2<T> operator*(const Vector2<T>& vector, PetscReal scalar);

template<typename T>
Vector2<T> operator*(PetscReal scalar, const Vector2<T>& vector);


static constexpr PetscInt Vector3_dim = 3;

template<typename T>
struct Vector3 {
  static constexpr Vector3 null{0, 0, 0};
  static constexpr Vector3 orth_x{1, 0, 0};
  static constexpr Vector3 orth_y{0, 1, 0};
  static constexpr Vector3 orth_z{0, 0, 1};

  T x;
  T y;
  T z;

  constexpr Vector3()
    : x{0}, y{0}, z{0} {}

  constexpr Vector3(const T& _value)
    : x{_value}, y{_value}, z{_value} {}

  constexpr Vector3(const T& _x, const T& _y, const T& _z)
    : x{_x}, y{_y}, z{_z} {}

  constexpr Vector3(T _vector[Vector3_dim])
    : x{_vector[X]}, y{_vector[Y]}, z{_vector[Z]} {}

  Vector3& operator+=(const Vector3& other);
  Vector3& operator-=(const Vector3& other);
  Vector3& operator*=(PetscReal scalar);
  Vector3& operator/=(PetscReal scalar);

  Vector3 operator+(const Vector3& other) const;
  Vector3 operator-(const Vector3& other) const;
  Vector3 operator/(PetscReal scalar) const;
  Vector3 operator+() const;
  Vector3 operator-() const;

  Vector3& normalize();
  Vector3 normalized() const;
  Vector3 cross(const Vector3& other) const;
  Vector2<T> squeeze_along(Axis axis) const;

  T dot(const Vector3& other) const;
  T square() const;
  PetscReal length() const;
};

template<typename T>
Vector3<T> operator*(const Vector3<T>& vector, PetscReal scalar);

template<typename T>
Vector3<T> operator*(PetscReal scalar, const Vector3<T>& vector);

#include "vector_classes.inl"

#endif  // SRC_VECTORS_VECTOR_CLASSES_H
