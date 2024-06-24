//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#ifdef MOOSE_GPU_SCOPE
#include "GPUHeader.h"
#endif

#include "MooseTypes.h"

template <typename T>
class GPUScalar
{
private:
  // Reference to the scalar value
  const T & _reference_value;
  // Copied scalar value
  T _copy_value;

public:
  GPUScalar() = delete;
  GPUScalar(const T & value) : _reference_value(value) {}

#ifdef MOOSE_GPU_SCOPE
  KOKKOS_FUNCTION GPUScalar(const GPUScalar<T> & object) : _reference_value(object._reference_value)
  {
    KOKKOS_IF_ON_HOST(_copy_value = object._reference_value;)
    KOKKOS_IF_ON_DEVICE(_copy_value = object._copy_value;)
  }
  KOKKOS_FUNCTION operator T() const
  {
    KOKKOS_IF_ON_HOST(return _reference_value;)
    KOKKOS_IF_ON_DEVICE(return _copy_value;)
  }
#endif
};

using GPUPostprocessorValue = GPUScalar<PostprocessorValue>;
