//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "GPUMaterialPropertyStorage.h"

#include "FEProblemBase.h"
#include "MaterialBase.h"

GPUMaterialPropertyBase &
GPUMaterialPropertyStorage::addGPUProperty(const std::string & prop_name,
                                           const std::type_info & type,
                                           const unsigned int state,
                                           const MaterialBase * const declarer,
                                           const std::vector<unsigned int> & dims,
                                           std::shared_ptr<GPUMaterialPropertyBase> shell,
                                           const bool bnd)
{
  auto & mesh = _problem.mesh();
  auto & assembly = _problem.gpuAssembly();

  auto id = addProperty(prop_name, type, state, declarer);

  if (!_properties.count(prop_name))
    _properties[prop_name] = shell;

  _properties[prop_name]->allocate(mesh, assembly, declarer->blockIDs(), prop_name, id, dims, bnd);

  return *_properties[prop_name];
}

GPUMaterialPropertyBase &
GPUMaterialPropertyStorage::getGPUProperty(std::string prop_name)
{
  return *_properties[prop_name];
}

bool
GPUMaterialPropertyStorage::haveGPUProperty(std::string prop_name)
{
  return _properties.count(prop_name);
}

void
dataStore(std::ostream & stream, GPUMaterialPropertyStorage & storage, void * context)
{
}

void
dataLoad(std::istream & stream, GPUMaterialPropertyStorage & storage, void * context)
{
}
