//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "MaterialPropertyStorage.h"

class FEProblemBase;
class GPUMaterialPropertyStorage;

void dataStore(std::ostream & stream, GPUMaterialPropertyStorage & record, void * context);
void dataLoad(std::istream & stream, GPUMaterialPropertyStorage & record, void * context);

class GPUMaterialPropertyStorage : public MaterialPropertyStorage
{
  friend void dataStore(std::ostream &, GPUMaterialPropertyStorage &, void *);
  friend void dataLoad(std::istream &, GPUMaterialPropertyStorage &, void *);

public:
  GPUMaterialPropertyStorage(MaterialPropertyRegistry & registry, FEProblemBase & problem)
    : MaterialPropertyStorage(registry, problem)
  {
  }

#ifdef MOOSE_GPU_SCOPE
public:
  // Add a GPU material property to the storage
  GPUMaterialPropertyBase & addGPUProperty(const std::string & prop_name,
                                           const std::type_info & type,
                                           const unsigned int state,
                                           const MaterialBase * const declarer,
                                           const std::vector<unsigned int> & dims,
                                           std::shared_ptr<GPUMaterialPropertyBase> shell,
                                           const bool bnd);
  // Get a GPU material property
  GPUMaterialPropertyBase & getGPUProperty(std::string prop_name);
  // Check whether a GPU material property exists
  bool haveGPUProperty(std::string prop_name);
#endif

private:
  // GPU material properties
  std::map<std::string, std::shared_ptr<GPUMaterialPropertyBase>> _properties;
};
