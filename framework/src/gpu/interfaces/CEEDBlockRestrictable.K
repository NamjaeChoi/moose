//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifndef MOOSE_IGNORE_LIBCEED

#include "BlockRestrictable.h"

void
BlockRestrictable::initializeCEEDBlockRestrictable(
    const CEEDAssembly & assembly, const std::set<MooseVariableFieldBase *> & variables)
{
  auto blocks = blockRestricted() ? &blockIDs() : nullptr;
  auto boundaries = !_boundary_ids.empty() ? &_boundary_ids : nullptr;

  _rstr.createElementRestriction(assembly, variables, blocks, boundaries);
  _rstr.createQuadratureRestriction(assembly, variables, blocks, boundaries);
}

#endif
