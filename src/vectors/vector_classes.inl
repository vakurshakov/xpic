#include "vector_classes.h"

#include <cmath>  // for sqrt

#define VECTOR_DEFAULT_CONSTRUCTORS_IMPL(VEC_T, N)                                                \
  template<typename T> constexpr VEC_T<T>::VEC_T() : data{REP##N(0)} {}                           \
  template<typename T> constexpr VEC_T<T>::VEC_T(const T& v) : data{REP##N(v)} {}                 \
  template<typename T> constexpr VEC_T<T>::VEC_T(REP##N##_N(const T& x)) : data{REP##N##_N(x)} {} \
  template<typename T> constexpr VEC_T<T>::VEC_T(const T v[VEC_T##_dim]) : data{REP##N##_A(v)} {} \

VECTOR_DEFAULT_CONSTRUCTORS_IMPL(Vector2, 2)
VECTOR_DEFAULT_CONSTRUCTORS_IMPL(Vector3, 3)
VECTOR_DEFAULT_CONSTRUCTORS_IMPL(Vector4, 4)

#define PLUS +
#define COMMA ,
#define SEMICOLON ;

#define VEC_OP2(A, B, OP, SEP) A[0] OP B[0] SEP A[1] OP B[1] SEP
#define VEC_OP3(A, B, OP, SEP) A[0] OP B[0] SEP A[1] OP B[1] SEP A[2] OP B[2] SEP
#define VEC_OP4(A, B, OP, SEP) A[0] OP B[0] SEP A[1] OP B[1] SEP A[2] OP B[2] SEP A[3] OP B[3] SEP

#define SCALAR_OP2(A, B, OP, SEP) A[0] OP B SEP A[1] OP B SEP
#define SCALAR_OP3(A, B, OP, SEP) A[0] OP B SEP A[1] OP B SEP A[2] OP B SEP
#define SCALAR_OP4(A, B, OP, SEP) A[0] OP B SEP A[1] OP B SEP A[2] OP B SEP A[3] OP B SEP


#define RETURN_REF_VEC_OP(VEC_T, OP, N)                    \
  template<typename T>                                     \
  VEC_T<T>& VEC_T<T>::operator OP(const VEC_T<T>& other) { \
    VEC_OP##N(data, other, OP, SEMICOLON)                  \
    return *this;                                          \
  }                                                        \

RETURN_REF_VEC_OP(Vector2, +=, 2)
RETURN_REF_VEC_OP(Vector3, +=, 3)
RETURN_REF_VEC_OP(Vector4, +=, 4)

RETURN_REF_VEC_OP(Vector2, -=, 2)
RETURN_REF_VEC_OP(Vector3, -=, 3)
RETURN_REF_VEC_OP(Vector4, -=, 4)


#define RETURN_REF_SCALAR_OP(VEC_T, OP, N)                         \
  template<typename T> VEC_T<T>& VEC_T<T>::operator OP(T scalar) { \
    SCALAR_OP##N(data, scalar, *=, SEMICOLON)                      \
    return *this;                                                  \
  }                                                                \

RETURN_REF_SCALAR_OP(Vector2, *=, 2)
RETURN_REF_SCALAR_OP(Vector3, *=, 3)
RETURN_REF_SCALAR_OP(Vector4, *=, 4)


#define RETURN_NEW_VEC_OP(VEC_T, OP, N)                         \
  template<typename T>                                          \
  VEC_T<T> VEC_T<T>::operator OP(const VEC_T<T>& other) const { \
    return {                                                    \
      VEC_OP##N(data, other, OP, COMMA)                         \
    };                                                          \
  }                                                             \

RETURN_NEW_VEC_OP(Vector2, +, 2)
RETURN_NEW_VEC_OP(Vector3, +, 3)
RETURN_NEW_VEC_OP(Vector4, +, 4)

RETURN_NEW_VEC_OP(Vector2, -, 2)
RETURN_NEW_VEC_OP(Vector3, -, 3)
RETURN_NEW_VEC_OP(Vector4, -, 4)


#define RETURN_NEW_SCALAR_DIV(VEC_T, N)                  \
  template<typename T>                                   \
  VEC_T<PetscReal> VEC_T<T>::operator/(T scalar) const { \
    return {                                             \
      SCALAR_OP##N((PetscReal)data, scalar, /, COMMA)    \
    };                                                   \
  }                                                      \

RETURN_NEW_SCALAR_DIV(Vector2, 2)
RETURN_NEW_SCALAR_DIV(Vector3, 3)
RETURN_NEW_SCALAR_DIV(Vector4, 4)


#define RETURN_NEW_UNARY_OP(VEC_T, OP, N)                  \
  template<typename T>                                     \
  VEC_T<T> VEC_T<T>::operator OP() const {                 \
    return {                                               \
      SCALAR_OP##N(OP data, /* none */, /* none */, COMMA) \
    };                                                     \
  }                                                        \

RETURN_NEW_UNARY_OP(Vector2, +, 2)
RETURN_NEW_UNARY_OP(Vector3, +, 3)
RETURN_NEW_UNARY_OP(Vector4, +, 4)

RETURN_NEW_UNARY_OP(Vector2, -, 2)
RETURN_NEW_UNARY_OP(Vector3, -, 3)
RETURN_NEW_UNARY_OP(Vector4, -, 4)


#define RETURN_NEW_NORMALIZED(VEC_T, N)            \
  template<typename T>                             \
  VEC_T<PetscReal> VEC_T<T>::normalized() const {  \
    PetscReal len = length();                      \
    return {                                       \
      SCALAR_OP##N((PetscReal)data, len, /, COMMA) \
    };                                             \
  }                                                \

RETURN_NEW_NORMALIZED(Vector2, 2)
RETURN_NEW_NORMALIZED(Vector3, 3)
RETURN_NEW_NORMALIZED(Vector4, 4)


#define RETURN_VALUE_DOT(VEC_T, N)               \
  template<typename T>                           \
  T VEC_T<T>::dot(const VEC_T<T>& other) const { \
    return                                       \
      VEC_OP##N(data, other, *, PLUS) 0;         \
  }                                              \

RETURN_VALUE_DOT(Vector2, 2)
RETURN_VALUE_DOT(Vector3, 3)
RETURN_VALUE_DOT(Vector4, 4)


#define RETURN_VALUE_SQUARE(VEC_T) \
  template<typename T>             \
  T VEC_T<T>::square() const {     \
    return dot(*this);             \
  }                                \

RETURN_VALUE_SQUARE(Vector2)
RETURN_VALUE_SQUARE(Vector3)
RETURN_VALUE_SQUARE(Vector4)


#define RETURN_REAL_LENGTH(VEC_T)      \
  template<typename T>                 \
  PetscReal VEC_T<T>::length() const { \
    return sqrt((PetscReal)square());  \
  }                                    \

RETURN_REAL_LENGTH(Vector2)
RETURN_REAL_LENGTH(Vector3)
RETURN_REAL_LENGTH(Vector4)


#define RETURN_NEW_MULTIPLICATION(VEC_T, N)              \
  template<typename T>                                   \
  VEC_T<T> operator*(const VEC_T<T>& vector, T scalar) { \
    return {                                             \
      SCALAR_OP##N(vector, scalar, *, COMMA)             \
    };                                                   \
  }                                                      \
  \
  template<typename T>                                   \
  VEC_T<T> operator*(T scalar, const VEC_T<T>& vector) { \
    return {                                             \
      SCALAR_OP##N(vector, scalar, *, COMMA)             \
    };                                                   \
  }                                                      \

RETURN_NEW_MULTIPLICATION(Vector2, 2)
RETURN_NEW_MULTIPLICATION(Vector3, 3)
RETURN_NEW_MULTIPLICATION(Vector4, 4)


template<typename T>
Vector3<T> Vector3<T>::cross(const Vector3<T>& other) const {
  return {
    + (data[Y] * other[Z] - data[Z] * other[Y]),
    - (data[X] * other[Z] - data[Z] * other[X]),
    + (data[X] * other[Y] - data[Y] * other[X]),
  };
}

template<typename T>
Vector2<T> Vector3<T>::squeeze_along(Axis axis) const {
  switch (axis) {
    case Axis::X: return {data[Y], data[Z]};
    case Axis::Y: return {data[X], data[Z]};
    case Axis::Z: // fallthrough
    default:
      return {data[X], data[Y]};
  }
}
