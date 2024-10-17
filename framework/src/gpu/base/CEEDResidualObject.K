//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifndef MOOSE_IGNORE_LIBCEED

#include "CEEDResidualObject.h"

InputParameters
CEEDResidualObject::validParams()
{
  auto params = ResidualObject::validParams();

  params.addParam<bool>("use_displaced_mesh",
                        false,
                        "Whether or not this object should use the "
                        "displaced mesh for computation. Note that in "
                        "the case this is true but no displacements "
                        "are provided in the Mesh block the "
                        "undisplaced mesh will still be used.");
  params.addParamNamesToGroup("use_displaced_mesh", "Advanced");

  params.addCoupledVar("displacements", "The displacements");
  params.addPrivateParam<bool>("_ceed_object", true);

  return params;
}

CEEDResidualObject::CEEDResidualObject(const InputParameters & parameters, bool nodal)
  : ResidualObject(parameters, nodal),
    _ceed(getMooseApp().getCeed()),
    _assembly(_fe_problem.ceedAssembly()),
    _systems(_fe_problem.getCEEDSystems())
{
}

CEEDResidualObject::CEEDResidualObject(const CEEDResidualObject & object)
  : ResidualObject(object.parameters(), object.isNodal(), false),
    _ceed(object._ceed),
    _assembly(object._assembly),
    _systems(object._systems)
{
}

#endif
