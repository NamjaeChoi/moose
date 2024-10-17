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

#include "CEEDResidualObject.h"
#include "BlockRestrictable.h"
#include "MaterialPropertyInterface.h"

class CEEDKernelBase : public CEEDResidualObject,
                       public BlockRestrictable,
                       public CoupleableMooseVariableDependencyIntermediateInterface,
                       public MaterialPropertyInterface
{
public:
  static InputParameters validParams();

  CEEDKernelBase(const InputParameters & parameters);
  CEEDKernelBase(const CEEDKernelBase & object);
};

#endif
