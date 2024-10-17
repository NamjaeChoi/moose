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

#include "GPUHeader.h"
#include "GPUArray.h"
#include "CEEDAssembly.h"
#include "CEEDSystem.h"

#include "MooseVariableBase.h"
#include "ResidualObject.h"

class CEEDResidualObject : public ResidualObject
{
public:
  static InputParameters validParams();

  CEEDResidualObject(const InputParameters & parameters, bool nodal = false);
  CEEDResidualObject(const CEEDResidualObject & object);

  virtual void computeOffDiagJacobian(unsigned int) override final
  {
    mooseError("computeOffDiagJacobian() is not used for CEED residual objects.");
  }
  virtual void computeResidualAndJacobian() override final
  {
    computeResidual();
    computeJacobian();
  }

protected:
  // CEED logical device
  Ceed _ceed;
  // CEED assembly
  CEEDAssembly & _assembly;
  // CEED systems
  GPUArray<CEEDSystem> & _systems;
};

#endif
