#ifndef SRC_VECTORS_VECTOR_UTILS_H
#define SRC_VECTORS_VECTOR_UTILS_H

#include "src/vectors/vector3.h"
#include "src/vectors/vector4.h"

template<typename T, typename V>
Vector4<T> vector_cast(const Vector3<V>& vector) {
  return {
    (T)vector[X],
    (T)vector[Y],
    (T)vector[Z],
    (T)0,
  };
}

template<typename T>
Vector4<T> vector_cast(const Vector3<T>& vector) {
  return {
    vector[X],
    vector[Y],
    vector[Z],
    0,
  };
}

template<typename T, typename V>
Vector3<T> vector_cast(const Vector4<V>& vector) {
  return {
    (T)vector[X],
    (T)vector[Y],
    (T)vector[Z],
  };
}

template<typename T>
Vector3<T> vector_cast(const Vector4<T>& vector) {
  return {
    vector[X],
    vector[Y],
    vector[Z],
  };
}

template<typename T, typename V>
Vector4<T> vector_cast(const Vector4<V>& vector) {
  return {
    (T)vector[X],
    (T)vector[Y],
    (T)vector[Z],
    (T)vector[C],
  };
}

template<typename T, typename V>
Vector3<T> vector_cast(const Vector3<V>& vector) {
  return {
    (T)vector[X],
    (T)vector[Y],
    (T)vector[Z],
  };
}

#endif // SRC_VECTORS_VECTOR_UTILS_H
