//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifndef MOOSE_IGNORE_LIBCEED

#include "FEProblemBase.h"
#include "NonlinearSystemBase.h"
#include "AuxiliarySystem.h"
#include "MaterialBase.h"

void
FEProblemBase::initCEED()
{
  // Initialize CEED systems

  unsigned int max_system_number = 0;

  for (unsigned int s = 0; s < numNonlinearSystems(); ++s)
    max_system_number = std::max(max_system_number, getNonlinearSystemBase(s).number());

  max_system_number = std::max(max_system_number, getAuxiliarySystem().number());

  _ceed_systems.create(max_system_number + 1);

  for (unsigned int s = 0; s < numNonlinearSystems(); ++s)
  {
    auto & nl = getNonlinearSystemBase(s);
    _ceed_systems[nl.number()].init(nl, _ceed_assembly);
  }

  auto & aux = getAuxiliarySystem();
  _ceed_systems[aux.number()].init(aux, _ceed_assembly);
}

#endif
