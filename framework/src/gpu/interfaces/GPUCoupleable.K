//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "Coupleable.h"
#include "SystemBase.h"
#include "FEProblemBase.h"

GPUVariable
Coupleable::coupledGPUVar(const std::string & var_name)
{
  GPUVariable variable;

  const auto coupled = isCoupled(var_name);
  const auto components = coupledComponents(var_name);

  variable._coupled = coupled;
  variable._components = components;

  if (coupled)
  {
    variable._vars.create(components);

    for (unsigned int comp = 0; comp < components; ++comp)
    {
      const auto * const var = getVarHelper<MooseVariableField<Real>>(var_name, comp);

      variable._sys = var->sys().number();
      variable._vars[comp] = var->number();
    }

    variable._vars.copy();

    variable._tag = _c_fe_problem.getVectorTagID(Moose::SOLUTION_TAG);
  }
  else
  {
    variable._default_values.create(components);

    for (unsigned int comp = 0; comp < components; ++comp)
      variable._default_values[comp] = _c_parameters.defaultCoupledValue(var_name, comp);

    variable._default_values.copy();
  }

  return variable;
}

GPUVariable
Coupleable::coupledGPUTagVar(const std::string & var_name, const std::string & tag_name)
{
  GPUVariable variable = coupledGPUVar(var_name);

  if (!_c_parameters.isParamValid(tag_name))
    mooseError("Tag name parameter '", tag_name, "' is invalid");

  TagName tagname = _c_parameters.get<TagName>(tag_name);
  if (!_c_fe_problem.vectorTagExists(tagname))
    mooseError("Tagged vector with tag name '", tagname, "' does not exist");

  variable._tag = _c_fe_problem.getVectorTagID(tagname);

  return variable;
}
