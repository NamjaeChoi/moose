//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "GPUMaterialBase.h"

MaterialBase::MaterialBase(const MaterialBase & object)
  : MooseObject(object.parameters(), false),
    BlockRestrictable(object),
    BoundaryRestrictable(object),
    SetupInterface(&object, false),
    MooseVariableDependencyInterface(&object, false),
    ScalarCoupleable(&object, false),
    FunctionInterface(&object, false),
    DistributionInterface(&object, false),
    UserObjectInterface(&object, false),
    TransientInterface(&object, false),
    PostprocessorInterface(&object, false),
    VectorPostprocessorInterface(&object, false),
    DependencyResolverInterface(false),
    Restartable(&object, "MaterialBases", false),
    MeshChangedInterface(object.parameters(), false),
    OutputInterface(object.parameters(), false, false),
    RandomInterface(object.parameters(), object._fe_problem, object._tid, false, false),
    ElementIDInterface(&object, false),
    GeometricSearchInterface(&object, false),
    ADFunctorInterface(&object, false),
    SolutionInvalidInterface(&object, false),
    _subproblem(object._subproblem),
    _fe_problem(object._fe_problem),
    _tid(object._tid),
    _assembly(object._assembly),
    _coord(object._coord),
    _normals(object._normals),
    _mesh(object._mesh),
    _coord_sys(object._coord_sys),
    _compute(object._compute),
    _has_stateful_property(object._has_stateful_property),
    _declare_suffix(object._declare_suffix),
    _force_stateful_init(object._force_stateful_init)
{
}

InputParameters
GPUMaterialBase::validParams()
{
  InputParameters params = MaterialBase::validParams();

  params.addPrivateParam<bool>("_gpu_object", true);

  return params;
}

GPUMaterialBase::GPUMaterialBase(const InputParameters & parameters)
  : MaterialBase(parameters),
    _assembly_ref(_fe_problem.gpuAssembly()),
    _systems_ref(_fe_problem.getGPUSystems())
{
}

GPUMaterialBase::GPUMaterialBase(const GPUMaterialBase & object)
  : MaterialBase(object), _assembly_ref(object._assembly_ref), _systems_ref(object._systems_ref)
{
  _assembly = _assembly_ref;
  _systems = _systems_ref;
}
