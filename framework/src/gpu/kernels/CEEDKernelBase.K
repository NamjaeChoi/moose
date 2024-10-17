//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifndef MOOSE_IGNORE_LIBCEED

#include "CEEDKernelBase.h"

#include "MooseVariableFE.h"
#include "Problem.h"
#include "SubProblem.h"
#include "SystemBase.h"
#include "NonlinearSystem.h"

InputParameters
CEEDKernelBase::validParams()
{
  auto params = CEEDResidualObject::validParams();
  params += BlockRestrictable::validParams();
  params += MaterialPropertyInterface::validParams();

  // Kernels always couple within their element
  params.addRelationshipManager("ElementSideNeighborLayers",
                                Moose::RelationshipManagerType::COUPLING,
                                [](const InputParameters &, InputParameters & rm_params)
                                { rm_params.set<unsigned short>("layers") = 0; });
  return params;
}

CEEDKernelBase::CEEDKernelBase(const InputParameters & parameters)
  : CEEDResidualObject(parameters),
    BlockRestrictable(this),
    CoupleableMooseVariableDependencyIntermediateInterface(this, false),
    MaterialPropertyInterface(this, blockIDs(), Moose::EMPTY_BOUNDARY_IDS)
{
}

CEEDKernelBase::CEEDKernelBase(const CEEDKernelBase & object)
  : CEEDResidualObject(object),
    BlockRestrictable(&object, false),
    CoupleableMooseVariableDependencyIntermediateInterface(&object, false, false, false),
    MaterialPropertyInterface(&object, blockIDs(), Moose::EMPTY_BOUNDARY_IDS, false)
{
}

#endif
