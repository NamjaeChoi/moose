//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUResidualObject.h"

#include "BlockRestrictable.h"
#include "MaterialPropertyInterface.h"

class GPUKernelBase : public GPUResidualObject,
                      public BlockRestrictable,
                      public CoupleableMooseVariableDependencyIntermediateInterface,
                      public MaterialPropertyInterface
{
public:
  static InputParameters validParams();

  GPUKernelBase(const InputParameters & parameters);
  GPUKernelBase(const GPUKernelBase & object);
};
