//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUDirichletBCBase.h"

class GPUDirichletBC final : public GPUDirichletBCBase<GPUDirichletBC>
{
public:
  static InputParameters validParams();

  GPUDirichletBC(const InputParameters & parameters);

  KOKKOS_FUNCTION Real computeValue(const ResidualNodalDatum &) const { return _value; }

protected:
  const Real _value;
};
