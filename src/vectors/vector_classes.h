#ifndef SRC_VECTORS_VECTOR_CLASSES_H
#define SRC_VECTORS_VECTOR_CLASSES_H

#include <cstdint>
#include <iostream>

enum Axis : std::uint8_t {
  X = 0,
  Y = 1,
  Z = 2
};

template<typename T>
struct vector2 {
  static constexpr std::uint8_t dim = 2;

  static constexpr vector2 null{0, 0};
  static constexpr vector2 orth_x{1, 0};
  static constexpr vector2 orth_y{0, 1};

  union {
    double vector[dim];
    struct {
      double x;
      double y;
    };
  };

  constexpr vector2()
    : x{0}, y{0} {}

  constexpr vector2(const T& _value)
    : x{_value}, y{_value} {}

  constexpr vector2(const T& _x, const T& _y)
    : x{_x}, y{_y} {}

  constexpr vector2(T _vector[dim])
    : x{_vector[X]}, y{_vector[Y]} {}

  vector2& operator+=(const vector2& other);
  vector2& operator-=(const vector2& other);
  vector2& operator*=(double scalar);
  vector2& operator/=(double scalar);

  vector2 operator+(const vector2& other) const;
  vector2 operator-(const vector2& other) const;
  vector2 operator/(double scalar) const;
  vector2 operator+() const;
  vector2 operator-() const;

  vector2& normalize();
  vector2 normalized() const;
  T dot(const vector2& other) const;
  T square() const;
  T length() const;
};

template<typename T>
vector2<T> operator*(const vector2<T>& vector, double scalar);

template<typename T>
vector2<T> operator*(double scalar, const vector2<T>& vector);

template<typename T>
std::ostream& operator<<(std::ostream& out, vector2<T>& vector);


template<typename T>
struct vector3 {
  static constexpr std::uint8_t dim = 3;

  static constexpr vector3 null{0, 0, 0};
  static constexpr vector3 orth_x{1, 0, 0};
  static constexpr vector3 orth_y{0, 1, 0};
  static constexpr vector3 orth_z{0, 0, 1};

  union {
    double vector[dim];
    struct {
      double x;
      double y;
      double z;
    };
  };

  constexpr vector3()
    : x{0}, y{0}, z{0} {}

  constexpr vector3(const T& _value)
    : x{_value}, y{_value}, z{_value} {}

  constexpr vector3(const T& _x, const T& _y, const T& _z)
    : x{_x}, y{_y}, z{_z} {}

  constexpr vector3(T _vector[dim])
    : x{_vector[X]}, y{_vector[Y]}, z{_vector[Z]} {}

  vector3& operator+=(const vector3& other);
  vector3& operator-=(const vector3& other);
  vector3& operator*=(double scalar);
  vector3& operator/=(double scalar);

  vector3 operator+(const vector3& other) const;
  vector3 operator-(const vector3& other) const;
  vector3 operator/(double scalar) const;
  vector3 operator+() const;
  vector3 operator-() const;

  vector3& normalize();
  vector3 normalized() const;
  vector3 cross(const vector3& other) const;
  vector2<T> squeeze_along(Axis axis) const;

  T dot(const vector3& other) const;
  T square() const;
  T length() const;
};

template<typename T>
vector3<T> operator*(const vector3<T>& vector, double scalar);

template<typename T>
vector3<T> operator*(double scalar, const vector3<T>& vector);

template<typename T>
std::ostream& operator<<(std::ostream& out, vector3<T>& vector);

#include "vector_classes.inl"

#endif  // SRC_VECTORS_VECTOR_CLASSES_H
