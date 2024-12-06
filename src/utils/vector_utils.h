#ifndef SRC_VECTORS_VECTOR_UTILS_H
#define SRC_VECTORS_VECTOR_UTILS_H

#include "src/utils/vector3.h"
#include "src/utils/vector4.h"

template<typename T, typename V>
Vector4<T> vector_cast(const Vector3<V>& vector)
{
  return Vector4{
    static_cast<T>(vector[X]),
    static_cast<T>(vector[Y]),
    static_cast<T>(vector[Z]),
    static_cast<T>(0),
  };
}

template<typename T>
Vector4<T> vector_cast(const Vector3<T>& vector)
{
  return Vector4{
    vector[X],
    vector[Y],
    vector[Z],
    0,
  };
}

template<typename T, typename V>
Vector3<T> vector_cast(const Vector4<V>& vector)
{
  return Vector3{
    static_cast<T>(vector[X]),
    static_cast<T>(vector[Y]),
    static_cast<T>(vector[Z]),
  };
}

template<typename T>
Vector3<T> vector_cast(const Vector4<T>& vector)
{
  return Vector3{
    vector[X],
    vector[Y],
    vector[Z],
  };
}

template<typename T, typename V>
Vector4<T> vector_cast(const Vector4<V>& vector)
{
  return Vector4{
    static_cast<T>(vector[X]),
    static_cast<T>(vector[Y]),
    static_cast<T>(vector[Z]),
    static_cast<T>(vector[C]),
  };
}

template<typename T, typename V>
Vector3<T> vector_cast(const Vector3<V>& vector)
{
  return Vector3{
    static_cast<T>(vector[X]),
    static_cast<T>(vector[Y]),
    static_cast<T>(vector[Z]),
  };
}

#endif  // SRC_VECTORS_VECTOR_UTILS_H
