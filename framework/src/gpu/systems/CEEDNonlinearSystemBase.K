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
#include "NonlinearSystemBase.h"
#include "MooseVariableFieldBase.h"

void
NonlinearSystemBase::addCEEDKernel(const std::string & kernel_name,
                                   const std::string & name,
                                   InputParameters & parameters)
{
  // Create the kernel object via the factory and add to warehouse
  std::shared_ptr<CEEDKernelBase> kernel =
      _factory.create<CEEDKernelBase>(kernel_name, name, parameters);
  _ceed_kernels.addObject(kernel, 0);
  postAddResidualObject(*kernel);
}

void
NonlinearSystemBase::computeCEEDResidual(const std::set<TagID> & tags)
{
  TIME_SECTION("computeCEEDResidual", 1);

  auto & systems = _fe_problem.getCEEDSystems();

  systems[number()].setActiveResidualTags(tags);

  // Resolve dependencies

  std::set<MooseVariableFieldBase *> needed_moose_vars;
  std::set<TagID> needed_fe_var_vector_tags;
  std::unordered_set<unsigned int> needed_mat_props;

  for (auto tag : _fe_problem.getVectorTags(Moose::VECTOR_TAG_SOLUTION))
    if (hasVector(tag._id))
      needed_fe_var_vector_tags.insert(tag._id);

  for (auto block : _fe_problem.mesh().meshSubdomains())
  {
    _ceed_kernels.updateBlockVariableDependency(block, needed_moose_vars);
    _ceed_kernels.updateBlockFEVariableCoupledVectorTagDependency(block, needed_fe_var_vector_tags);
    _ceed_kernels.updateBlockMatPropDependency(block, needed_mat_props);
  }

  // Project variables to quadature points

  for (auto & system : systems)
    if (system.initialized())
    {
      system.setActiveVariables(needed_moose_vars);
      system.setActiveVariableTags(needed_fe_var_vector_tags);

      {
        TIME_SECTION("CEEDCopy", 1);
        system.sync(GPUMemcpyKind::HOST_TO_DEVICE);
      }
      {
        TIME_SECTION("CEEDProject", 1);
        system.project();
      }
    }

  systems.copy();

  {
    TIME_SECTION("CEEDKernel", 1);

    // Compute kernels

    for (auto kernel : _ceed_kernels.getVectorTagsObjectWarehouse(tags, 0).getActiveObjects())
      kernel->computeResidual();

    Kokkos::fence();
  }

  // Copy back residual

  {
    TIME_SECTION("CEEDCopy", 1);
    systems[number()].sync(GPUMemcpyKind::DEVICE_TO_HOST);
  }

  // Clear

  systems[number()].clearActiveResidualTags();

  for (auto & system : systems)
    if (system.initialized())
    {
      system.clearActiveVariables();
      system.clearActiveVariableTags();
    }
}

#endif
