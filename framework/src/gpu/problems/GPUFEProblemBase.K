//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "GPUTypes.h"

#include "FEProblemBase.h"
#include "NonlinearSystemBase.h"
#include "AuxiliarySystem.h"
#include "MaterialBase.h"

void
FEProblemBase::initGPU()
{
  // Initialize GPU assembly

  _gpu_assembly.initElement();

  if (hasNeighborCoupling())
    _gpu_assembly.initNeighbor();

  // Initialize GPU systems

  unsigned int max_system_number = 0;

  for (unsigned int s = 0; s < numNonlinearSystems(); ++s)
    max_system_number = std::max(max_system_number, getNonlinearSystemBase(s).number());

  max_system_number = std::max(max_system_number, getAuxiliarySystem().number());

  _gpu_systems.create(max_system_number + 1);

  for (unsigned int s = 0; s < numNonlinearSystems(); ++s)
  {
    auto & nl = getNonlinearSystemBase(s);
    _gpu_systems[nl.number()].init(nl, _gpu_assembly);
  }

  auto & aux = getAuxiliarySystem();
  _gpu_systems[aux.number()].init(aux, _gpu_assembly);
}

void
FEProblemBase::prepareGPUMaterials(
    const std::unordered_set<unsigned int> & consumer_needed_mat_props)
{
  std::unordered_set<unsigned int> needed_mat_props;

  for (auto block : mesh().meshSubdomains())
    _gpu_materials.updateBlockMatPropDependency(block, needed_mat_props);

  for (auto boundary : mesh().meshBoundaryIds())
    _gpu_materials.updateBoundaryMatPropDependency(boundary, needed_mat_props);

  needed_mat_props.insert(consumer_needed_mat_props.begin(), consumer_needed_mat_props.end());

  setActiveMaterialProperties(needed_mat_props, 0);
}

void
FEProblemBase::reinitGPUMaterials()
{
  if (hasActiveMaterialProperties(0))
    for (auto & material : _gpu_materials.getActiveObjects())
      material->computeProperties();

  Kokkos::fence();
}
