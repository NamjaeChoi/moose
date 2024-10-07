//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifndef MOOSE_IGNORE_LIBCEED

#pragma once

#include <memory>

// NVCC invokes compile error due to conflicting built-in and PETSc complex operators
// so this preprocessor should be defined
#define PETSC_SKIP_CXX_COMPLEX_FIX 1

#include <ceed.h>

// Class-wrapped CEED objects that has destructor
#define CEED_OBJECT(name, ceed, destroy)                                                           \
  class name                                                                                       \
  {                                                                                                \
  private:                                                                                         \
    std::shared_ptr<ceed> _ceed;                                                                   \
                                                                                                   \
  public:                                                                                          \
    operator bool() { return *_ceed; }                                                             \
    operator ceed &() { return *_ceed; }                                                           \
    ceed * operator&() { return _ceed.get(); }                                                     \
    name()                                                                                         \
    {                                                                                              \
      _ceed = std::make_shared<ceed>();                                                            \
      *_ceed = NULL;                                                                               \
    }                                                                                              \
    ~name()                                                                                        \
    {                                                                                              \
      if (_ceed.use_count() == 1)                                                                  \
        destroy(_ceed.get());                                                                      \
    }                                                                                              \
  }

CEED_OBJECT(CEED, Ceed, CeedDestroy);
CEED_OBJECT(CEEDVector, CeedVector, CeedVectorDestroy);
CEED_OBJECT(CEEDBasis, CeedBasis, CeedBasisDestroy);
CEED_OBJECT(CEEDElemRestriction, CeedElemRestriction, CeedElemRestrictionDestroy);
CEED_OBJECT(CEEDOperator, CeedOperator, CeedOperatorDestroy);
CEED_OBJECT(CEEDQFunction, CeedQFunction, CeedQFunctionDestroy);
CEED_OBJECT(CEEDQFunctionContext, CeedQFunctionContext, CeedQFunctionContextDestroy);

using CEEDQFunctionPointer = int (*)(void *,
                                     const CeedInt,
                                     const CeedScalar * const *,
                                     CeedScalar * const *);

#define CEED_FUNCTION KOKKOS_FUNCTION

// libMesh uses an old Boost library and causes a build error due to an outdated CUDA
// preprocessor which should be manually undefined after including CUDA runtime header
#undef __CUDACC_VER__

// MOOSE includes
#include "MooseError.h"
#include "MooseUtils.h"

#endif
