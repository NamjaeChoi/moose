//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

// NVCC invokes compile error due to conflicting built-in and PETSc complex operators
// so this preprocessor should be defined
#define PETSC_SKIP_CXX_COMPLEX_FIX 1

// Kokkos headers
#include "Kokkos_Core.hpp"
#include "Kokkos_UnorderedMap.hpp"
#include "Kokkos_StdAlgorithms.hpp"

// libMesh uses an old Boost library and causes a build error due to an outdated CUDA
// preprocessor which should be manually undefined after including CUDA runtime header
#undef __CUDACC_VER__

// MOOSE includes
#include "MooseError.h"
#include "MooseUtils.h"

// Architecture-dependent execution spaces
#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#endif
#ifdef KOKKOS_ENABLE_HIP
#define MemSpace Kokkos::HIPSpace
#endif
#ifdef KOKKOS_ENABLE_SYCL
#define MemSpace Kokkos::Experimental::SYCL
#endif

// Convenient mirror class
template <typename T>
class GPUMirror
{
public:
  GPUMirror() {}
  GPUMirror(const T & object) { create(object); }
  ~GPUMirror() { free(); }

  KOKKOS_FUNCTION T * get() { return _mirror; }

  void create(const T & object)
  {
    _mirror = static_cast<T *>(Kokkos::kokkos_malloc(sizeof(T)));

    Kokkos::Impl::DeepCopy<MemSpace, Kokkos::HostSpace>(_mirror, &object, sizeof(T));
    Kokkos::fence();
  }
  void free() { Kokkos::kokkos_free(_mirror); }

private:
  T * _mirror = nullptr;
};
