//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GPUTypes.h"

#include "MaterialBase.h"

class GPUMaterialBase : public MaterialBase
{
public:
  static InputParameters validParams();

  GPUMaterialBase(const InputParameters & parameters);
  GPUMaterialBase(const GPUMaterialBase & object);

  // Unused for GPUs
  virtual void subdomainSetup() override final {}

  // GPU function tags
  struct ElementLoop
  {
  };
  struct SideLoop
  {
  };
  struct NeighborLoop
  {
  };

  // Declare GPU material property
  template <typename T, unsigned int dimension = 0>
  GPUMaterialProperty<T, dimension> declareProperty(const std::string & name,
                                                    const std::vector<unsigned int> dims = {})
  {
    std::string prop_name = name;
    if (_pars.have_parameter<MaterialPropertyName>(name))
      prop_name = _pars.get<MaterialPropertyName>(name);

    return declarePropertyByName<T, dimension>(prop_name, dims);
  }
  template <typename T, unsigned int dimension = 0>
  GPUMaterialProperty<T, dimension> declarePropertyByName(const std::string & prop_name,
                                                          const std::vector<unsigned int> dims = {})
  {
    return declareGPUProperty<T, dimension>(prop_name, dims);
  }

private:
  template <typename T, unsigned int dimension>
  GPUMaterialProperty<T, dimension> declareGPUProperty(const std::string & prop_name,
                                                       const std::vector<unsigned int> dims = {})
  {
    if (dimension > 4)
      mooseError("Up to four-dimensional GPU material properties are allowed.");

    if (dims.size() != dimension)
      mooseError("The declared GPU material property dimension (",
                 dimension,
                 ")",
                 " and the provided number of dimensions (",
                 dims.size(),
                 ") should match.");

    const auto prop_name_modified =
        _declare_suffix.empty()
            ? prop_name
            : MooseUtils::join(std::vector<std::string>({prop_name, _declare_suffix}), "_");

    auto prop = materialData().declareGPUProperty<T, dimension>(
        prop_name_modified, dims, *this, isBoundaryMaterial());

    registerPropName(prop_name_modified, false, 0);

    return prop;
  }

private:
  // Copy of GPU assembly (for device access)
  GPUAssembly _assembly;
  // Reference to GPU assembly (for host access)
  GPUAssembly & _assembly_ref;
  // Copy of GPU systems (for device access)
  GPUArray<GPUSystem> _systems;
  // Reference to GPU systems (for host access)
  GPUArray<GPUSystem> & _systems_ref;

protected:
  KOKKOS_FUNCTION const GPUAssembly & assembly() const
  {
    KOKKOS_IF_ON_HOST(return _assembly_ref;)
    KOKKOS_IF_ON_DEVICE(return _assembly;)
  }
  KOKKOS_FUNCTION const GPUArray<GPUSystem> & systems() const
  {
    KOKKOS_IF_ON_HOST(return _systems_ref;)
    KOKKOS_IF_ON_DEVICE(return _systems;)
  }
};
