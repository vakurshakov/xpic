#include "vector_classes.h"

#include <cmath>  // for sqrt

template<typename T>
vector2<T>& vector2<T>::operator+=(const vector2<T>& other) {
  x += other.x;
  y += other.y;
  return *this;
}

template<typename T>
vector2<T>& vector2<T>::operator-=(const vector2<T>& other) {
  x -= other.x;
  y -= other.y;
  return *this;
}

template<typename T>
vector2<T>& vector2<T>::operator*=(double scalar) {
  x *= scalar;
  y *= scalar;
  return *this;
}

template<typename T>
vector2<T>& vector2<T>::operator/=(double scalar) {
  x /= scalar;
  y /= scalar;
  return *this;
}


template<typename T>
vector2<T> vector2<T>::operator+(const vector2<T>& other) const {
  return {
    x + other.x,
    y + other.y,
  };
}

template<typename T>
vector2<T> vector2<T>::operator-(const vector2<T>& other) const {
  return {
    x - other.x,
    y - other.y,
  };
}

template<typename T>
vector2<T> vector2<T>::operator/(double scalar) const {
  return {
    x / scalar,
    y / scalar,
  };
}

template<typename T>
vector2<T> vector2<T>::operator+() const {
  return {
    x,
    y,
  };
}

template<typename T>
vector2<T> vector2<T>::operator-() const {
  return {
    -x,
    -y,
  };
}


template<typename T>
vector2<T>& vector2<T>::normalize() {
  T tmp = length();
  x /= tmp;
  y /= tmp;
  return *this;
}

template<typename T>
vector2<T> vector2<T>::normalized() const {
  T tmp = length();
  return {
    x / tmp,
    y / tmp,
  };
}

template<typename T>
T vector2<T>::dot(const vector2<T>& other) const {
  return
    x * other.x +
    y * other.y;
}

template<typename T>
T vector2<T>::square() const {
  return dot(*this);
}

template<typename T>
T vector2<T>::length() const {
  return sqrt(square());
}


template<typename T>
vector2<T> operator*(const vector2<T>& vector, double scalar) {
  return {
    vector.x * scalar,
    vector.y * scalar,
  };
}

template<typename T>
vector2<T> operator*(double scalar, const vector2<T>& vector) {
  return {
    vector.x * scalar,
    vector.y * scalar,
  };
}

template<typename T>
std::ostream& operator<<(std::ostream& out, vector2<T>& vector) {
  out << v.x << " " << v.y;
  return out;
}


template<typename T>
vector3<T>& vector3<T>::operator+=(const vector3<T>& other) {
  x += other.x;
  y += other.y;
  z += other.z;
  return *this;
}

template<typename T>
vector3<T>& vector3<T>::operator-=(const vector3<T>& other) {
  x -= other.x;
  y -= other.y;
  z -= other.z;
  return *this;
}

template<typename T>
vector3<T>& vector3<T>::operator*=(double scalar) {
  x *= scalar;
  y *= scalar;
  z *= scalar;
  return *this;
}

template<typename T>
vector3<T>& vector3<T>::operator/=(double scalar) {
  x /= scalar;
  y /= scalar;
  z /= scalar;
  return *this;
}


template<typename T>
vector3<T> vector3<T>::operator+(const vector3<T>& other) const {
  return {
    x + other.x,
    y + other.y,
    z + other.z,
  };
}

template<typename T>
vector3<T> vector3<T>::operator-(const vector3<T>& other) const {
  return {
    x - other.x,
    y - other.y,
    z - other.z,
  };
}

template<typename T>
vector3<T> vector3<T>::operator/(double scalar) const {
  return {
    x / scalar,
    y / scalar,
    z / scalar,
  };
}

template<typename T>
vector3<T> vector3<T>::operator+() const {
  return {
    x,
    y,
    z,
  };
}

template<typename T>
vector3<T> vector3<T>::operator-() const {
  return {
    -x,
    -y,
    -z,
  };
}


template<typename T>
vector3<T>& vector3<T>::normalize() {
  T tmp = length();
  x /= tmp;
  y /= tmp;
  z /= tmp;
  return *this
}

template<typename T>
vector3<T> vector3<T>::normalized() const {
  T tmp = length();
  return {
    x / tmp,
    y / tmp,
    z / tmp,
  };
}

template<typename T>
vector3<T> vector3<T>::cross(const vector3<T>& other) const {
  return {
    + (y * other.z - z * other.y),
    - (x * other.z - z * other.x),
    + (x * other.y - y * other.x),
  };
}

template<typename T>
vector2<T> vector3<T>::squeeze_along(Axis axis) const {
  switch (axis) {
    case Axis::X: return {y, z};
    case Axis::Y: return {x, z};
    case Axis::Z: // fallthrough
    default:
      return {x, y};
  }
}

template<typename T>
T vector3<T>::dot(const vector3<T>& other) const {
  return
    x * other.x +
    y * other.y +
    z * other.z;
}

template<typename T>
T vector3<T>::square() const {
  return dot(*this);
}

template<typename T>
T vector3<T>::length() const {
  return sqrt(square());
}


template<typename T>
vector3<T> operator*(const vector3<T>& vector, double scalar) {
  return {
    vector.x * scalar,
    vector.y * scalar,
    vector.z * scalar,
  };
}

template<typename T>
vector3<T> operator*(double scalar, const vector3<T>& vector) {
  return {
    vector.x * scalar,
    vector.y * scalar,
    vector.z * scalar,
  };
}

template<typename T>
std::ostream& operator<<(std::ostream& out, vector3<T>& vector) {
  out << v.x << " " << v.y << " " << v.z;
  return out;
}
