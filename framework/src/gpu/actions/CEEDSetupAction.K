//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifndef MOOSE_IGNORE_LIBCEED

#include "CEEDSetupAction.h"

#include "FEProblemBase.h"
#include "NonlinearSystemBase.h"
#include "AuxiliarySystem.h"

registerMooseAction("MooseApp", CEEDSetupAction, "ready_to_init");

InputParameters
CEEDSetupAction::validParams()
{
  InputParameters params = Action::validParams();
  return params;
}

CEEDSetupAction::CEEDSetupAction(const InputParameters & parameters) : Action(parameters) {}

void
CEEDSetupAction::act()
{
  if (!_app.hasGPUs())
    return;

  // Construct GPU assembly

  auto & ceed_assembly = _problem->ceedAssembly();

  if (_current_task == "ready_to_init")
    ceed_assembly.init(*_problem);
}

#endif
