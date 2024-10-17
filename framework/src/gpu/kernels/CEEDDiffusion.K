//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifndef MOOSE_IGNORE_LIBCEED

#include "CEEDDiffusion.h"

registerMooseObjectAliased("MooseApp", CEEDDiffusionKernel, "CEEDDiffusion");

InputParameters
CEEDDiffusionKernel::validParams()
{
  InputParameters params = CEEDDiffusion::validParams();
  params.addClassDescription("The Laplacian operator ($-\\nabla \\cdot \\nabla u$), with the weak "
                             "form of $(\\nabla \\phi_i, \\nabla u_h)$.");
  return params;
}

CEEDDiffusionKernel::CEEDDiffusionKernel(const InputParameters & parameters)
  : CEEDDiffusion(parameters)
{
}

#endif
