#include "vector_classes.h"

#include <petscmath.h>  // for sqrt

template<typename T>
Vector2<T>& Vector2<T>::operator+=(const Vector2<T>& other) {
  x += other.x;
  y += other.y;
  return *this;
}

template<typename T>
Vector2<T>& Vector2<T>::operator-=(const Vector2<T>& other) {
  x -= other.x;
  y -= other.y;
  return *this;
}

template<typename T>
Vector2<T>& Vector2<T>::operator*=(PetscReal scalar) {
  x *= scalar;
  y *= scalar;
  return *this;
}

template<typename T>
Vector2<T>& Vector2<T>::operator/=(PetscReal scalar) {
  x /= scalar;
  y /= scalar;
  return *this;
}


template<typename T>
Vector2<T> Vector2<T>::operator+(const Vector2<T>& other) const {
  return {
    x + other.x,
    y + other.y,
  };
}

template<typename T>
Vector2<T> Vector2<T>::operator-(const Vector2<T>& other) const {
  return {
    x - other.x,
    y - other.y,
  };
}

template<typename T>
Vector2<T> Vector2<T>::operator/(PetscReal scalar) const {
  return {
    x / scalar,
    y / scalar,
  };
}

template<typename T>
Vector2<T> Vector2<T>::operator+() const {
  return {
    x,
    y,
  };
}

template<typename T>
Vector2<T> Vector2<T>::operator-() const {
  return {
    -x,
    -y,
  };
}


template<typename T>
Vector2<PetscReal> Vector2<T>::normalized() const {
  PetscReal len = length();
  return {
    x / len,
    y / len,
  };
}

template<typename T>
T Vector2<T>::dot(const Vector2<T>& other) const {
  return
    x * other.x +
    y * other.y;
}

template<typename T>
T Vector2<T>::square() const {
  return dot(*this);
}

template<typename T>
PetscReal Vector2<T>::length() const {
  return PetscSqrtReal(static_cast<PetscReal>(square()));
}


template<typename T>
Vector2<T> operator*(const Vector2<T>& vector, PetscReal scalar) {
  return {
    vector.x * scalar,
    vector.y * scalar,
  };
}

template<typename T>
Vector2<T> operator*(PetscReal scalar, const Vector2<T>& vector) {
  return {
    vector.x * scalar,
    vector.y * scalar,
  };
}


template<typename T>
Vector3<T>& Vector3<T>::operator+=(const Vector3<T>& other) {
  x += other.x;
  y += other.y;
  z += other.z;
  return *this;
}

template<typename T>
Vector3<T>& Vector3<T>::operator-=(const Vector3<T>& other) {
  x -= other.x;
  y -= other.y;
  z -= other.z;
  return *this;
}

template<typename T>
Vector3<T>& Vector3<T>::operator*=(PetscReal scalar) {
  x *= scalar;
  y *= scalar;
  z *= scalar;
  return *this;
}

template<typename T>
Vector3<T>& Vector3<T>::operator/=(PetscReal scalar) {
  x /= scalar;
  y /= scalar;
  z /= scalar;
  return *this;
}


template<typename T>
Vector3<T> Vector3<T>::operator+(const Vector3<T>& other) const {
  return {
    x + other.x,
    y + other.y,
    z + other.z,
  };
}

template<typename T>
Vector3<T> Vector3<T>::operator-(const Vector3<T>& other) const {
  return {
    x - other.x,
    y - other.y,
    z - other.z,
  };
}

template<typename T>
Vector3<T> Vector3<T>::operator/(PetscReal scalar) const {
  return {
    x / scalar,
    y / scalar,
    z / scalar,
  };
}

template<typename T>
Vector3<T> Vector3<T>::operator+() const {
  return {
    x,
    y,
    z,
  };
}

template<typename T>
Vector3<T> Vector3<T>::operator-() const {
  return {
    -x,
    -y,
    -z,
  };
}


template<typename T>
Vector3<PetscReal> Vector3<T>::normalized() const {
  PetscReal len = length();
  return {
    x / len,
    y / len,
    z / len,
  };
}

template<typename T>
Vector3<T> Vector3<T>::cross(const Vector3<T>& other) const {
  return {
    + (y * other.z - z * other.y),
    - (x * other.z - z * other.x),
    + (x * other.y - y * other.x),
  };
}

template<typename T>
Vector2<T> Vector3<T>::squeeze_along(Axis axis) const {
  switch (axis) {
    case Axis::X: return {y, z};
    case Axis::Y: return {x, z};
    case Axis::Z: // fallthrough
    default:
      return {x, y};
  }
}

template<typename T>
T Vector3<T>::dot(const Vector3<T>& other) const {
  return
    x * other.x +
    y * other.y +
    z * other.z;
}

template<typename T>
T Vector3<T>::square() const {
  return dot(*this);
}

template<typename T>
PetscReal Vector3<T>::length() const {
  return PetscSqrtReal(static_cast<PetscReal>(square()));
}


template<typename T>
Vector3<T> operator*(const Vector3<T>& vector, PetscReal scalar) {
  return {
    vector.x * scalar,
    vector.y * scalar,
    vector.z * scalar,
  };
}

template<typename T>
Vector3<T> operator*(PetscReal scalar, const Vector3<T>& vector) {
  return {
    vector.x * scalar,
    vector.y * scalar,
    vector.z * scalar,
  };
}
