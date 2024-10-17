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

#include "CEEDKernel.h"

CEED_KERNEL_TEMPLATE(CEEDDiffusion, CEEDKernel, Kernel)
public:
static InputParameters
validParams()
{
  InputParameters params = CEEDKernel<Kernel>::validParams();
  return params;
}

CEEDDiffusion(const InputParameters & parameters) : CEEDKernel<Kernel>(parameters) {}

CEED_FUNCTION CeedScalar
computeQpResidual(const CeedInt qp)
{
  printf("%d\n", qp);

  return 0;
}

CEED_KERNEL_TEMPLATE_END(CEEDDiffusion)

CEED_KERNEL(CEEDDiffusionKernel, CEEDDiffusion)
public:
static InputParameters validParams();

CEEDDiffusionKernel(const InputParameters & parameters);
CEED_KERNEL_END(CEEDDiffusionKernel)

#endif
