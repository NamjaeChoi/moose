//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "GPUNodalBCBase.h"

InputParameters
GPUNodalBCBase::validParams()
{
  auto params = GPUBoundaryCondition::validParams();

  return params;
}

GPUNodalBCBase::GPUNodalBCBase(const InputParameters & parameters)
  : GPUBoundaryCondition(parameters, true),
    CoupleableMooseVariableDependencyIntermediateInterface(this, true)
{
}

GPUNodalBCBase::GPUNodalBCBase(const GPUNodalBCBase & object)
  : GPUBoundaryCondition(object),
    CoupleableMooseVariableDependencyIntermediateInterface(&object, true, false, false)
{
}
