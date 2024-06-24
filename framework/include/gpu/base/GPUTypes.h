//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUArray.h"

#include "MooseError.h"
#include "MooseUtils.h"

#include "libmesh/tensor_tools.h"

struct Real3
{
  Real x, y, z;

#ifdef MOOSE_GPU_SCOPE
  KOKKOS_INLINE_FUNCTION Real3() : x(0), y(0), z(0) {}
  KOKKOS_INLINE_FUNCTION Real3(const Real & scalar) : x(scalar), y(scalar), z(scalar) {}
  KOKKOS_INLINE_FUNCTION Real3(const Real3 & vector) : x(vector.x), y(vector.y), z(vector.z) {}
  KOKKOS_INLINE_FUNCTION Real3(const Real & x, const Real & y, const Real & z) : x(x), y(y), z(z) {}

  KOKKOS_INLINE_FUNCTION void operator=(Real scalar)
  {
    x = scalar;
    y = scalar;
    z = scalar;
  }
  void operator=(const libMesh::TypeVector<Real> & vector)
  {
    x = vector(0);
#if LIBMESH_DIM > 1
    y = vector(1);
#else
    y = 0;
#endif
#if LIBMESH_DIM > 2
    z = vector(2);
#else
    z = 0;
#endif
  }
  KOKKOS_INLINE_FUNCTION void operator+=(Real scalar)
  {
    x += scalar;
    y += scalar;
    z += scalar;
  }
  KOKKOS_INLINE_FUNCTION void operator+=(Real3 vector)
  {
    x += vector.x;
    y += vector.y;
    z += vector.z;
  }
  KOKKOS_INLINE_FUNCTION void operator-=(Real scalar)
  {
    x -= scalar;
    y -= scalar;
    z -= scalar;
  }
  KOKKOS_INLINE_FUNCTION void operator-=(Real3 vector)
  {
    x -= vector.x;
    y -= vector.y;
    z -= vector.z;
  }
  KOKKOS_INLINE_FUNCTION void operator*=(Real scalar)
  {
    x *= scalar;
    y *= scalar;
    z *= scalar;
  }
#endif
};

struct Real33
{
  Real a[3][3];

#ifdef MOOSE_GPU_SCOPE
  KOKKOS_INLINE_FUNCTION Real33() { *this = 0; }
  KOKKOS_INLINE_FUNCTION Real33(const Real & scalar) { *this = scalar; }

  KOKKOS_INLINE_FUNCTION Real & operator()(unsigned int i, unsigned int j) { return a[i][j]; }
  KOKKOS_INLINE_FUNCTION const Real & operator()(unsigned int i, unsigned int j) const
  {
    return a[i][j];
  }
  KOKKOS_INLINE_FUNCTION void operator=(Real scalar)
  {
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        a[i][j] = scalar;
  }
#endif
};

#ifdef MOOSE_GPU_SCOPE
KOKKOS_INLINE_FUNCTION Real3
operator*(Real left, Real3 right)
{
  return {left * right.x, left * right.y, left * right.z};
}
KOKKOS_INLINE_FUNCTION Real3
operator*(Real3 left, Real right)
{
  return {left.x * right, left.y * right, left.z * right};
}
KOKKOS_INLINE_FUNCTION Real
operator*(Real3 left, Real3 right)
{
  return left.x * right.x + left.y * right.y + left.z * right.z;
}
KOKKOS_INLINE_FUNCTION Real3
operator*(Real33 left, Real3 right)
{
  return {left(0, 0) * right.x + left(0, 1) * right.y + left(0, 2) * right.z,
          left(1, 0) * right.x + left(1, 1) * right.y + left(1, 2) * right.z,
          left(2, 0) * right.x + left(2, 1) * right.y + left(2, 2) * right.z};
}
KOKKOS_INLINE_FUNCTION Real3
operator+(Real left, Real3 right)
{
  return {left + right.x, left + right.y, left + right.z};
}
KOKKOS_INLINE_FUNCTION Real3
operator+(Real3 left, Real right)
{
  return {left.x + right, left.y + right, left.z + right};
}
KOKKOS_INLINE_FUNCTION Real3
operator+(Real3 left, Real3 right)
{
  return {left.x + right.x, left.y + right.y, left.z + right.z};
}
KOKKOS_INLINE_FUNCTION Real3
operator-(Real left, Real3 right)
{
  return {left - right.x, left - right.y, left - right.z};
}
KOKKOS_INLINE_FUNCTION Real3
operator-(Real3 left, Real right)
{
  return {left.x - right, left.y - right, left.z - right};
}
KOKKOS_INLINE_FUNCTION Real3
operator-(Real3 left, Real3 right)
{
  return {left.x - right.x, left.y - right.y, left.z - right.z};
}
#endif
