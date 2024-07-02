//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "GPUResidualObject.h"

InputParameters
GPUResidualObject::validParams()
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

  params.addPrivateParam<bool>("_gpu_object", true);

  return params;
}

GPUResidualObject::GPUResidualObject(const InputParameters & parameters, bool nodal)
  : ResidualObject(parameters, nodal),
    _assembly_ref(_fe_problem.gpuAssembly()),
    _systems_ref(_fe_problem.getGPUSystems())
{
}

GPUResidualObject::GPUResidualObject(const GPUResidualObject & object)
  : ResidualObject(object.parameters(), object.isNodal(), false),
    _assembly_ref(object._assembly_ref),
    _systems_ref(object._systems_ref)
{
  _vector_tags = object.getVectorTags(VectorTagsKey());
  _matrix_tags = object.getMatrixTags(MatrixTagsKey());

  _assembly = _assembly_ref;
  _systems = _systems_ref;

  _gpu_var = object._gpu_var;
}

void
GPUResidualObject::setVariable(const MooseVariableBase & var)
{
  std::vector<unsigned int> vars;

  for (unsigned int i = 0; i < var.count(); ++i)
    vars.push_back(var.number() + i);

  _gpu_var._sys = var.sys().number();
  _gpu_var._vars = vars;
  _gpu_var._tag = _fe_problem.getVectorTagID(Moose::SOLUTION_TAG);
}
