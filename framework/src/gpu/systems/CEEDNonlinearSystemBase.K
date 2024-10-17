//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "CEEDKernelBase.h"
#include "NonlinearSystemBase.h"
#include "MooseVariableFieldBase.h"

void
NonlinearSystemBase::addCEEDKernel(const std::string & kernel_name,
                                   const std::string & name,
                                   InputParameters & parameters)
{
  // Create the kernel object via the factory and add to warehouse
  std::shared_ptr<CEEDKernelBase> kernel =
      _factory.create<CEEDKernelBase>(kernel_name, name, parameters);
  _ceed_kernels.addObject(kernel, 0);
  postAddResidualObject(*kernel);
}

