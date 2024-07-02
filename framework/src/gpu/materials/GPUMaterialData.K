//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "GPUMaterialPropertyStorage.h"

#include "MaterialData.h"

GPUMaterialPropertyBase &
MaterialData::addGPUPropertyHelper(const std::string & prop_name,
                                   const std::type_info & type,
                                   const unsigned int state,
                                   const MaterialBase * const declarer,
                                   const std::vector<unsigned int> & dims,
                                   std::shared_ptr<GPUMaterialPropertyBase> shell,
                                   const bool bnd) const
{
  auto & storage = static_cast<GPUMaterialPropertyStorage &>(_storage);

  return storage.addGPUProperty(prop_name, type, state, declarer, dims, shell, bnd);
}

GPUMaterialPropertyBase &
MaterialData::getGPUPropertyHelper(const std::string & prop_name, const unsigned int state) const
{
  auto & storage = static_cast<GPUMaterialPropertyStorage &>(_storage);

  return storage.getGPUProperty(prop_name);
}

bool
MaterialData::haveGPUPropertyHelper(const std::string & prop_name) const
{
  auto & storage = static_cast<GPUMaterialPropertyStorage &>(_storage);

  return storage.haveGPUProperty(prop_name);
}
