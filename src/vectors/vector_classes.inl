#include "vector_classes.h"

#include <cmath>  // for sqrt

#define VECTOR_DEFAULT_CONSTRUCTORS_IMPL(N)                                                                  \
  template<typename T> constexpr Vector##N<T>::Vector##N() : data{REP##N(T(0))} {}                           \
  template<typename T> constexpr Vector##N<T>::Vector##N(const T& v) : data{REP##N(v)} {}                    \
  template<typename T> constexpr Vector##N<T>::Vector##N(REP##N##_N(const T& x)) : data{REP##N##_N(x)} {}    \
  template<typename T> constexpr Vector##N<T>::Vector##N(const T v[Vector##N::dim]) : data{REP##N##_A(v)} {} \

VECTOR_DEFAULT_CONSTRUCTORS_IMPL(3)
VECTOR_DEFAULT_CONSTRUCTORS_IMPL(4)


#define PLUS +
#define COMMA ,
#define SEMICOLON ;

#define VEC_OP2(A, B, OP, SEP) A[0] OP B[0] SEP A[1] OP B[1]
#define VEC_OP3(A, B, OP, SEP) A[0] OP B[0] SEP A[1] OP B[1] SEP A[2] OP B[2]
#define VEC_OP4(A, B, OP, SEP) A[0] OP B[0] SEP A[1] OP B[1] SEP A[2] OP B[2] SEP A[3] OP B[3]

#define VEC_FUNC2(A, B, FUNC, SEP)  FUNC(A[0], B[0]) SEP FUNC(A[1], B[1])
#define VEC_FUNC3(A, B, FUNC, SEP)  FUNC(A[0], B[0]) SEP FUNC(A[1], B[1]) SEP FUNC(A[2], B[2])
#define VEC_FUNC4(A, B, FUNC, SEP)  FUNC(A[0], B[0]) SEP FUNC(A[1], B[1]) SEP FUNC(A[2], B[2]) SEP FUNC(A[3], B[3])

#define SCALAR_OP2(A, B, OP, SEP) A[0] OP B SEP A[1] OP B
#define SCALAR_OP3(A, B, OP, SEP) A[0] OP B SEP A[1] OP B SEP A[2] OP B
#define SCALAR_OP4(A, B, OP, SEP) A[0] OP B SEP A[1] OP B SEP A[2] OP B SEP A[3] OP B


#define RETURN_REF_VEC_OP(OP, N)                                       \
  template<typename T>                                                 \
  Vector##N<T>& Vector##N<T>::operator OP(const Vector##N<T>& other) { \
    VEC_OP##N(data, other, OP, SEMICOLON);                             \
    return *this;                                                      \
  }                                                                    \

RETURN_REF_VEC_OP(+=, 3)
RETURN_REF_VEC_OP(-=, 3)
RETURN_REF_VEC_OP(*=, 3)

RETURN_REF_VEC_OP(+=, 4)
RETURN_REF_VEC_OP(-=, 4)
RETURN_REF_VEC_OP(*=, 4)


#define RETURN_REF_SCALAR_OP(OP, N)                   \
  template<typename T>                                \
  Vector##N<T>& Vector##N<T>::operator OP(T scalar) { \
    SCALAR_OP##N(data, scalar, *=, SEMICOLON);        \
    return *this;                                     \
  }                                                   \

RETURN_REF_SCALAR_OP(*=, 3)
RETURN_REF_SCALAR_OP(*=, 4)


#define RETURN_NEW_VEC_OP(OP, N)                                            \
  template<typename T>                                                      \
  Vector##N<T> Vector##N<T>::operator OP(const Vector##N<T>& other) const { \
    return { VEC_OP##N(data, other, OP, COMMA) };                           \
  }                                                                         \

RETURN_NEW_VEC_OP(+, 3)
RETURN_NEW_VEC_OP(-, 3)
RETURN_NEW_VEC_OP(*, 3)

RETURN_NEW_VEC_OP(+, 4)
RETURN_NEW_VEC_OP(-, 4)
RETURN_NEW_VEC_OP(*, 4)


#define RETURN_NEW_SCALAR_DIV(N)                                 \
  template<typename T>                                           \
  Vector##N<PetscReal> Vector##N<T>::operator/(T scalar) const { \
    return { SCALAR_OP##N((PetscReal)data, scalar, /, COMMA) };  \
  }                                                              \

RETURN_NEW_SCALAR_DIV(3)
RETURN_NEW_SCALAR_DIV(4)


#define RETURN_NEW_NORMALIZED(N)                          \
  template<typename T>                                    \
  Vector##N<PetscReal> Vector##N<T>::normalized() const { \
    PetscReal len = length();                             \
    return operator/(len);                                \
  }                                                       \

RETURN_NEW_NORMALIZED(3)
RETURN_NEW_NORMALIZED(4)


#define RETURN_VALUE_DOT(N)                              \
  template<typename T>                                   \
  T Vector##N<T>::dot(const Vector##N<T>& other) const { \
    return VEC_OP##N(data, other, *, PLUS);              \
  }                                                      \

RETURN_VALUE_DOT(3)
RETURN_VALUE_DOT(4)


#define RETURN_VALUE_SQUARE(N)                          \
  template<typename T>                                  \
  T Vector##N<T>::square() const { return dot(*this); } \

RETURN_VALUE_SQUARE(3)
RETURN_VALUE_SQUARE(4)


#define RETURN_REAL_LENGTH(N)              \
  template<typename T>                     \
  PetscReal Vector##N<T>::length() const { \
    return sqrt((PetscReal)square());      \
  }                                        \

RETURN_REAL_LENGTH(3)
RETURN_REAL_LENGTH(4)


#define SWAP3(A) A[X], A[Z]
#define SWAP4(A) A[X], A[Z]

#define UPDATE_SWAP_ORDER(N)        \
  template<typename T>              \
  void Vector##N<T>::swap_order() { \
    std::swap(SWAP##N(data));       \
  }                                 \

UPDATE_SWAP_ORDER(3)
UPDATE_SWAP_ORDER(4)


#define RETURN_NEW_MULTIPLICATION(N)                             \
  template<typename T>                                           \
  Vector##N<T> operator*(const Vector##N<T>& vector, T scalar) { \
    return { SCALAR_OP##N(vector, scalar, *, COMMA) };           \
  }                                                              \
  \
  template<typename T>                                           \
  Vector##N<T> operator*(T scalar, const Vector##N<T>& vector) { \
    return { SCALAR_OP##N(vector, scalar, *, COMMA) };           \
  }                                                              \

RETURN_NEW_MULTIPLICATION(3)
RETURN_NEW_MULTIPLICATION(4)


#define RETURN_NEW_MINMAX_COMPARISON(N)                                \
  template<typename T>                                                 \
  Vector##N<T> min(const Vector##N<T>& lhs, const Vector##N<T>& rhs) { \
    return { VEC_FUNC##N(lhs, rhs, std::min, COMMA) };                 \
  }                                                                    \
  \
  template<typename T>                                                 \
  Vector##N<T> max(const Vector##N<T>& lhs, const Vector##N<T>& rhs) { \
    return { VEC_FUNC##N(lhs, rhs, std::max, COMMA) };                 \
  }                                                                    \

RETURN_NEW_MINMAX_COMPARISON(3)
RETURN_NEW_MINMAX_COMPARISON(4)
